#pragma once

#include <torch/extension.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_start;
    size_t data_end;
};

class SafetensorsReader {
private:
    std::string filepath_;
    size_t header_size_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    int file_fd_ = -1;
    void* mmap_addr_ = nullptr;
    size_t file_size_ = 0;

    void parse_header(const std::string& json_str);

public:
    SafetensorsReader(const std::string& filepath);
    ~SafetensorsReader();

    bool has_tensor(const std::string& name) const;
    std::vector<std::string> get_tensor_names() const;
    TensorInfo get_tensor_info(const std::string& name) const;
    torch::Tensor load_tensor(const std::string& name, torch::Device device = torch::kCPU);
};

class ModelWeightIndex {
private:
    std::unordered_map<std::string, std::shared_ptr<SafetensorsReader>> readers_;
    std::unordered_map<std::string, std::string> tensor_to_file_;
    std::mutex mutex_;

public:
    ModelWeightIndex() = default;
    void load_model_dir(const std::string& model_dir);
    bool has_tensor(const std::string& tensor_name);
    torch::Tensor load_tensor(const std::string& tensor_name, torch::Device device = torch::kCPU);
};

extern ModelWeightIndex g_native_weight_index;
