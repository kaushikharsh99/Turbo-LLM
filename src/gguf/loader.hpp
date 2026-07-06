#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <memory>
#include <cstdint>

namespace turbo {

enum class TensorType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ2_S = 20,
    IQ3_S = 21,
    IQ4_NL = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    U8 = 28,
    U16 = 29,
    U32 = 30,
    U64 = 31,
    F8_E4M3 = 32,
    F8_E5M2 = 33,
};

struct GGUFTensor {
    std::string name;
    std::vector<int64_t> shape;
    TensorType type;
    uint64_t offset;          // relative to data block start
    uint64_t size_bytes;
    uint64_t absolute_offset; // absolute file offset of tensor data
};

struct GGUFMetadataValue {
    uint32_t type;
    std::vector<uint8_t> raw_data;
};

class GGUFLoader {
private:
    std::string filepath;
    std::ifstream file;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint64_t data_offset; // Absolute offset to binary data block start
    
    std::unordered_map<std::string, GGUFMetadataValue> metadata;
    std::unordered_map<std::string, GGUFTensor> tensors;
    std::vector<std::string> tensor_names;

public:
    GGUFLoader(const std::string& path);
    ~GGUFLoader();

    bool load_header_and_metadata();
    
    const std::unordered_map<std::string, GGUFMetadataValue>& get_metadata() const { return metadata; }
    const std::unordered_map<std::string, GGUFTensor>& get_tensors() const { return tensors; }
    const std::vector<std::string>& get_tensor_names() const { return tensor_names; }
    
    bool read_tensor_data(const GGUFTensor& tensor, void* buffer);
    bool read_tensor_data(const std::string& name, void* buffer);
    
    bool get_metadata_string(const std::string& key, std::string& val) const;
    bool get_metadata_uint32(const std::string& key, uint32_t& val) const;
    bool get_metadata_uint64(const std::string& key, uint64_t& val) const;
    bool get_metadata_float(const std::string& key, float& val) const;
};

} // namespace turbo
