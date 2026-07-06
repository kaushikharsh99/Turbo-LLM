#include "safetensors_reader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

SafetensorsReader::SafetensorsReader(const std::string& filepath) : filepath_(filepath) {
    struct stat st;
    if (stat(filepath.c_str(), &st) != 0) {
        throw std::runtime_error("SafetensorsReader: Cannot stat file " + filepath);
    }
    file_size_ = st.st_size;

    file_fd_ = open(filepath.c_str(), O_RDONLY);
    if (file_fd_ < 0) {
        throw std::runtime_error("SafetensorsReader: Cannot open file " + filepath);
    }

    mmap_addr_ = mmap(NULL, file_size_, PROT_READ, MAP_SHARED, file_fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        close(file_fd_);
        file_fd_ = -1;
        throw std::runtime_error("SafetensorsReader: Cannot mmap file " + filepath);
    }

    // Read uint64 header size
    uint64_t header_size_le = 0;
    std::memcpy(&header_size_le, mmap_addr_, sizeof(uint64_t));
    header_size_ = static_cast<size_t>(header_size_le);

    if (8 + header_size_ > file_size_) {
        munmap(mmap_addr_, file_size_);
        close(file_fd_);
        throw std::runtime_error("SafetensorsReader: Header size out of bounds in " + filepath);
    }

    std::string json_str(static_cast<const char*>(mmap_addr_) + 8, header_size_);
    parse_header(json_str);
}

SafetensorsReader::~SafetensorsReader() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, file_size_);
    }
    if (file_fd_ >= 0) {
        close(file_fd_);
    }
}

void SafetensorsReader::parse_header(const std::string& json_str) {
    // Simple, lightweight JSON parser for safetensors header entries:
    // "tensor_name": { "dtype": "F8_E4M3", "shape": [1408, 2048], "data_offsets": [begin, end] }
    size_t pos = 0;
    while ((pos = json_str.find('"', pos)) != std::string::npos) {
        size_t end_name = json_str.find('"', pos + 1);
        if (end_name == std::string::npos) break;

        std::string name = json_str.substr(pos + 1, end_name - pos - 1);
        if (name == "__metadata__") {
            pos = end_name + 1;
            continue;
        }

        size_t brace_start = json_str.find('{', end_name);
        if (brace_start == std::string::npos) break;
        size_t brace_end = json_str.find('}', brace_start);
        if (brace_end == std::string::npos) break;

        std::string block = json_str.substr(brace_start, brace_end - brace_start + 1);

        TensorInfo info;
        info.name = name;

        // Extract dtype
        size_t dtype_pos = block.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t v_start = block.find('"', dtype_pos + 7);
            if (v_start != std::string::npos) {
                size_t v_end = block.find('"', v_start + 1);
                if (v_end != std::string::npos) {
                    info.dtype = block.substr(v_start + 1, v_end - v_start - 1);
                }
            }
        }

        // Extract shape
        size_t shape_pos = block.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t s_start = block.find('[', shape_pos);
            size_t s_end = block.find(']', s_start);
            if (s_start != std::string::npos && s_end != std::string::npos) {
                std::string s_str = block.substr(s_start + 1, s_end - s_start - 1);
                std::stringstream ss(s_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) {
                        info.shape.push_back(std::stoll(item));
                    }
                }
            }
        }

        // Extract data_offsets
        size_t offset_pos = block.find("\"data_offsets\"");
        if (offset_pos != std::string::npos) {
            size_t o_start = block.find('[', offset_pos);
            size_t o_end = block.find(']', o_start);
            if (o_start != std::string::npos && o_end != std::string::npos) {
                std::string o_str = block.substr(o_start + 1, o_end - o_start - 1);
                size_t comma = o_str.find(',');
                if (comma != std::string::npos) {
                    info.data_start = std::stoull(o_str.substr(0, comma));
                    info.data_end = std::stoull(o_str.substr(comma + 1));
                }
            }
        }

        if (!info.name.empty() && info.data_end > info.data_start) {
            tensors_[info.name] = info;
        }

        pos = brace_end + 1;
    }
}

bool SafetensorsReader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> SafetensorsReader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

TensorInfo SafetensorsReader::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("SafetensorsReader: Tensor not found " + name);
    }
    return it->second;
}

torch::Tensor SafetensorsReader::load_tensor(const std::string& name, torch::Device device) {
    TensorInfo info = get_tensor_info(name);
    size_t offset_in_file = 8 + header_size_ + info.data_start;
    size_t byte_length = info.data_end - info.data_start;

    const char* raw_ptr = static_cast<const char*>(mmap_addr_) + offset_in_file;

    // Determine PyTorch dtype
    torch::ScalarType stype = torch::kFloat16;
    if (info.dtype == "F8_E4M3" || info.dtype == "FLOAT8_E4M3FN") {
        stype = torch::kFloat8_e4m3fn;
    } else if (info.dtype == "F16" || info.dtype == "FLOAT16") {
        stype = torch::kFloat16;
    } else if (info.dtype == "BF16" || info.dtype == "BFLOAT16") {
        stype = torch::kBFloat16;
    } else if (info.dtype == "F32" || info.dtype == "FLOAT32") {
        stype = torch::kFloat32;
    }

    // Allocate PyTorch CPU tensor and copy directly from mmap memory
    torch::TensorOptions options = torch::TensorOptions().dtype(stype).device(torch::kCPU);
    torch::Tensor cpu_tensor = torch::empty(info.shape, options);
    std::memcpy(cpu_tensor.data_ptr(), raw_ptr, byte_length);

    if (device.is_cuda()) {
        if (torch::cuda::is_available()) {
            cpu_tensor = cpu_tensor.pin_memory();
        }
        return cpu_tensor.to(device, /*non_blocking=*/true);
    }
    return cpu_tensor;
}

// Global weight index
ModelWeightIndex g_native_weight_index;

void ModelWeightIndex::load_model_dir(const std::string& model_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            std::string path = entry.path().string();
            try {
                auto reader = std::make_shared<SafetensorsReader>(path);
                readers_[path] = reader;
                for (const std::string& t_name : reader->get_tensor_names()) {
                    tensor_to_file_[t_name] = path;
                }
            } catch (const std::exception& e) {
                std::cerr << "Native safetensors index error on " << path << ": " << e.what() << "\n";
            }
        }
    }
}

bool ModelWeightIndex::has_tensor(const std::string& tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return tensor_to_file_.find(tensor_name) != tensor_to_file_.end();
}

torch::Tensor ModelWeightIndex::load_tensor(const std::string& tensor_name, torch::Device device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tensor_to_file_.find(tensor_name);
    if (it == tensor_to_file_.end()) {
        throw std::runtime_error("ModelWeightIndex: Tensor not found " + tensor_name);
    }
    return readers_[it->second]->load_tensor(tensor_name, device);
}
