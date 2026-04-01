#include "PageManager.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace tritondb::storage {

PageManager::PageManager(std::filesystem::path file_path) : file_path_(std::move(file_path)) {
    {
        std::ofstream create_if_missing(file_path_, std::ios::binary | std::ios::app);
        if (!create_if_missing.is_open()) {
            throw std::runtime_error("Failed to create page file: " + file_path_.string());
        }
    }

    file_.open(file_path_, std::ios::in | std::ios::out | std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open page file: " + file_path_.string());
    }
}

void PageManager::ensure_open() const {
    if (!file_.is_open()) {
        throw std::runtime_error("Page file is not open");
    }
}

std::uint64_t PageManager::page_count() const {
    std::error_code ec;
    const auto size = std::filesystem::file_size(file_path_, ec);
    if (ec) {
        throw std::runtime_error("Failed to inspect page file: " + file_path_.string() + ": " + ec.message());
    }
    return size / kPageSize;
}

PageId PageManager::alloc() {
    ensure_open();
    const PageId id = page_count();

    Page page;
    page.id = id;
    write(page);

    return id;
}

void PageManager::write(const Page& page) {
    ensure_open();

    const auto offset = static_cast<std::streamoff>(page.id * kPageSize);
    file_.clear();
    file_.seekp(offset);
    if (!file_.good()) {
        throw std::runtime_error("seekp failed while writing page");
    }
    file_.write(reinterpret_cast<const char*>(page.data.data()), kPageSize);
    file_.flush();

    if (!file_.good()) {
        throw std::runtime_error("write failed for page");
    }
}
Page PageManager::read(const PageId id) const {
    ensure_open();
    if (id >= page_count()) {
        throw std::runtime_error("Requested page does not exist");
    }
    Page page;
    page.id = id;
    const auto offset = static_cast<std::streamoff>(id * kPageSize);
    file_.clear();
    file_.seekg(offset);
    if (!file_.good()) {
        throw std::runtime_error("seekg failed while writing page");
    }
    file_.read(reinterpret_cast<char*>(page.data.data()), kPageSize);
    if (file_.gcount() != static_cast<std::streamsize>(kPageSize)) {
        throw std::runtime_error("read failed for page");
    }

    return page;
}

}   // namespace tritondb::storage