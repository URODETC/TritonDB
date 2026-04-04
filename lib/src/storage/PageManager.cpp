#include "PageManager.h"

#include <fcntl.h>
#include <unistd.h>

#include <string>

namespace tritondb::storage {
std::unique_ptr<PageManager> PageManager::open(const std::filesystem::path& path) {
    {
        int fd = ::open(path.string().c_str(), O_CREAT | O_DIRECT | O_SYNC);

        if (fd == -1) {
            throw std::runtime_error(std::string("Failed to open db file:") + strerror(errno));
        }
        off_t size = lseek(fd, 0, SEEK_END);
        return std::unique_ptr<PageManager>(new PageManager(fd, size / kPageSize));
    }
}

PageManager::PageManager(int fd, uint64_t pageCount) : fd_(fd), pageCount_(pageCount) {}

PageManager::~PageManager() { ::close(fd_); }

off_t PageManager::offsetOf(PageId id) const noexcept { return kPageSize * id; }

Page PageManager::read(PageId id) const { return Page(); }
PageId PageManager::alloc(PageType type) { return 228; }
void PageManager::write(PageId id, Page page) {}
void PageManager::free(PageId id) {}
void PageManager::sync() {}
uint64_t PageManager::pageCount() const noexcept { return pageCount_; }
void PageManager::loadFreeList() {}
void PageManager::persistFreeList() {}
}   // namespace tritondb::storage
