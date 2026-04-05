#include "PageManager.h"

#include <cerrno>
#include <cstring>
#include <format>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace tritondb::storage {

static void preadFull(int fd, void* buf, std::size_t size, off_t offset) {
    auto* ptr = static_cast<char*>(buf);
    std::size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::pread(fd, ptr, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::format("pread failed: {}", std::strerror(errno)));
        }
        if (n == 0) {
            throw std::runtime_error(
                std::format("pread: unexpected EOF at offset {}", offset));
        }
        ptr += n;
        offset += n;
        remaining -= static_cast<std::size_t>(n);
    }
}

static void pwriteFull(int fd, const void* buf, std::size_t size, off_t offset) {
    const auto* ptr = static_cast<const char*>(buf);
    std::size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::pwrite(fd, ptr, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::format("pwrite failed: {}", std::strerror(errno)));
        }
        ptr += n;
        offset += n;
        remaining -= static_cast<std::size_t>(n);
    }
}

void PageManager::serializeFreeListPage(const FreeListPage& flPage, Page& page) noexcept {
    page.clear(PageType::FreeList);

    auto* dst = page.data.data();

    std::memcpy(dst, &flPage.nextPageId, sizeof(PageId));
    dst += sizeof(PageId);

    auto count = static_cast<uint32_t>(flPage.ids.size());
    std::memcpy(dst, &count, sizeof(uint32_t));
    dst += sizeof(uint32_t);

    if (count > 0) {
        std::memcpy(dst, flPage.ids.data(), count * sizeof(PageId));
    }

    page.updateChecksum();
}

PageManager::FreeListPage PageManager::deserializeFreeListPage(const Page& page) {
    FreeListPage flp;

    const auto* src = page.data.data();

    std::memcpy(&flp.nextPageId, src, sizeof(PageId));
    src += sizeof(PageId);

    uint32_t count = 0;
    std::memcpy(&count, src, sizeof(uint32_t));
    src += sizeof(uint32_t);

    if (count > kFreeListPageCapacity) {
        throw std::runtime_error(
            std::format("FreeListPage: corrupt count {}", count));
    }

    flp.ids.resize(count);
    if (count > 0) {
        std::memcpy(flp.ids.data(), src, count * sizeof(PageId));
    }

    return flp;
}

PageManager::PageManager(int fd, uint64_t pageCount) : fd_(fd), pageCount_(pageCount) {}

PageManager::~PageManager() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

std::unique_ptr<PageManager> PageManager::open(const std::filesystem::path& path) {
    int fd = ::open(path.string().c_str(), O_CREAT | O_RDWR, 0664);
    if (fd < 0) {
        throw std::runtime_error(
            std::format("Cannot open '{}': {}", path.string(), std::strerror(errno)));
    }
    off_t size = lseek(fd, 0, SEEK_END);
    if (size < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("lseek failed: {}", std::strerror(errno)));
    }

    uint64_t page_count = static_cast<uint64_t>(size) / kPageSize;

    auto pm = std::unique_ptr<PageManager>(new PageManager(fd, page_count));

    if (page_count == 0) {
        pm->pageCount_ = 0;
        pm->freeListHead_ = FreeListPage{};
        PageId head_id = pm->extendFile();
        (void)head_id;
        pm->flushHead();
    }
    else {
        pm->loadHead();
    }
    return pm;
}

off_t PageManager::offsetOf(PageId id) noexcept {
    return static_cast<off_t>(id) * static_cast<off_t>(kPageSize);
}

PageId PageManager::extendFile() {
    PageId new_id = pageCount_++;
    Page p{};
    p.header.id = new_id;
    p.header.type = PageType::Free;
    p.updateChecksum();
    pwriteFull(fd_, &p, kPageSize, offsetOf(new_id));
    return new_id;
}

PageManager::FreeListPage PageManager::readFreeListPage(PageId flPageId) const {
    Page page{};
    preadFull(fd_, &page, kPageSize, offsetOf(flPageId));

    if (!page.verifyChecksum()) {
        throw std::runtime_error(
            std::format("FreeListPage {}: checksum mismatch", flPageId));
    }
    if (page.header.type != PageType::FreeList) {
        throw std::runtime_error(
            std::format("FreeListPage {}: wrong page type", flPageId));
    }

    return deserializeFreeListPage(page);
}

void PageManager::writeFreeListPage(PageId flPageId, const FreeListPage& flPage) {
    Page page{};
    page.header.id = flPageId;
    serializeFreeListPage(flPage, page);
    pwriteFull(fd_, &page, kPageSize, offsetOf(flPageId));
}

void PageManager::flushHead() {
    writeFreeListPage(kFreeListHeadPageId, freeListHead_);
}

void PageManager::loadHead() {
    freeListHead_ = readFreeListPage(kFreeListHeadPageId);
}

PageId PageManager::popFree() {
    if (freeListHead_.ids.empty()) {
        if (freeListHead_.nextPageId == kInvalidPageId) {
            return kInvalidPageId;
        }

        PageId next_id = freeListHead_.nextPageId;
        FreeListPage next_page = readFreeListPage(next_id);

        freeListHead_.ids = std::move(next_page.ids);
        freeListHead_.nextPageId = next_page.nextPageId;
        freeListHead_.ids.push_back(next_id);

        flushHead();
    }

    PageId id = freeListHead_.ids.back();
    freeListHead_.ids.pop_back();
    flushHead();
    return id;
}

void PageManager::pushFree(PageId id) {
    if (freeListHead_.ids.size() < kFreeListPageCapacity) {
        freeListHead_.ids.push_back(id);
        flushHead();
        return;
    }

    PageId new_fl_page_id = extendFile();

    FreeListPage new_fl_page;
    new_fl_page.nextPageId = freeListHead_.nextPageId;
    new_fl_page.ids = std::move(freeListHead_.ids);
    writeFreeListPage(new_fl_page_id, new_fl_page);

    freeListHead_.nextPageId = new_fl_page_id;
    freeListHead_.ids.clear();
    freeListHead_.ids.push_back(id);
    flushHead();
}

PageId PageManager::alloc(PageType type) {
    PageId id = popFree();

    if (id == kInvalidPageId) {
        id = extendFile();
    }

    Page p{};
    p.header.id = id;
    p.clear(type);
    p.header.id = id;
    p.updateChecksum();
    pwriteFull(fd_, &p, kPageSize, offsetOf(id));

    return id;
}
Page PageManager::read(PageId id) const {
    if (id >= pageCount_) {
        throw std::runtime_error(
            std::format("PageManager::read: invalid PageId {}", id));
    }

    Page p{};
    preadFull(fd_, &p, kPageSize, offsetOf(id));

    if (!p.verifyChecksum()) {
        throw std::runtime_error(
            std::format("PageManager::read: checksum mismatch for page {}", id));
    }

    return p;
}

void PageManager::write(PageId id, Page page) {
    if (id >= pageCount_) {
        throw std::runtime_error(
            std::format("PageManager::write: invalid PageId {}", id));
    }

    page.header.id = id;
    page.updateChecksum();
    pwriteFull(fd_, &page, kPageSize, offsetOf(id));
}

void PageManager::free(PageId id) {
    if (id == kFreeListHeadPageId) {
        throw std::runtime_error("PageManager::free: cannot free page 0");
    }
    if (id >= pageCount_) {
        throw std::runtime_error(
            std::format("PageManager::free: invalid PageId {}", id));
    }
    pushFree(id);
}

void PageManager::sync() {
    if (::fsync(fd_) < 0) {
        throw std::runtime_error(
            std::format("fsync failed: {}", std::strerror(errno)));
    }
}
uint64_t PageManager::pageCount() const noexcept { return pageCount_; }

uint64_t PageManager::freeCount() const noexcept {
    uint64_t total = freeListHead_.ids.size();
    PageId next_id = freeListHead_.nextPageId;

    while (next_id != kInvalidPageId) {
        try {
            FreeListPage flp = readFreeListPage(next_id);
            total += flp.ids.size();
            next_id = flp.nextPageId;
        } catch (...) {
            break;
        }
    }

    return total;
}
}   // namespace tritondb::storage
