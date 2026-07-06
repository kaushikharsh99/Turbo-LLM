#include "loader.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace turbo {

static uint64_t read_u64(std::ifstream& f) {
    uint64_t val;
    f.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

static uint32_t read_u32(std::ifstream& f) {
    uint32_t val;
    f.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

static std::string read_gguf_str(std::ifstream& f) {
    uint64_t len = read_u64(f);
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

static void skip_gguf_value(std::ifstream& f, uint32_t type) {
    switch (type) {
        case 0: f.seekg(1, std::ios::cur); break; // UINT8
        case 1: f.seekg(1, std::ios::cur); break; // INT8
        case 2: f.seekg(2, std::ios::cur); break; // UINT16
        case 3: f.seekg(2, std::ios::cur); break; // INT16
        case 4: f.seekg(4, std::ios::cur); break; // UINT32
        case 5: f.seekg(4, std::ios::cur); break; // INT32
        case 6: f.seekg(4, std::ios::cur); break; // FLOAT32
        case 7: f.seekg(1, std::ios::cur); break; // BOOL
        case 8: { // STRING
            uint64_t len = read_u64(f);
            f.seekg(len, std::ios::cur);
            break;
        }
        case 9: { // ARRAY
            uint32_t item_type = read_u32(f);
            uint64_t len = read_u64(f);
            for (uint64_t i = 0; i < len; ++i) {
                skip_gguf_value(f, item_type);
            }
            break;
        }
        case 10: f.seekg(8, std::ios::cur); break; // UINT64
        case 11: f.seekg(8, std::ios::cur); break; // INT64
        case 12: f.seekg(8, std::ios::cur); break; // FLOAT64
        default:
            throw std::runtime_error("Unknown GGUF metadata value type: " + std::to_string(type));
    }
}

static std::vector<uint8_t> read_gguf_value_bytes(std::ifstream& f, uint32_t type) {
    std::vector<uint8_t> bytes;
    auto start_pos = f.tellg();
    skip_gguf_value(f, type);
    auto end_pos = f.tellg();
    f.seekg(start_pos);
    bytes.resize(end_pos - start_pos);
    f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return bytes;
}

GGUFLoader::GGUFLoader(const std::string& path) : filepath(path) {}

GGUFLoader::~GGUFLoader() {
    if (file.is_open()) {
        file.close();
    }
}

bool GGUFLoader::load_header_and_metadata() {
    file.open(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open GGUF file: " << filepath << std::endl;
        return false;
    }
    
    // Read header
    uint32_t magic = read_u32(file);
    if (magic != 0x46554747) { // "GGUF" in little endian
        std::cerr << "Invalid GGUF magic: 0x" << std::hex << magic << std::dec << std::endl;
        return false;
    }
    
    version = read_u32(file);
    if (version != 2 && version != 3) {
        std::cerr << "Unsupported GGUF version: " << version << std::endl;
        return false;
    }
    
    tensor_count = read_u64(file);
    metadata_kv_count = read_u64(file);
    
    // Read metadata KV pairs
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string key = read_gguf_str(file);
        uint32_t val_type = read_u32(file);
        std::vector<uint8_t> raw_val = read_gguf_value_bytes(file, val_type);
        metadata[key] = {val_type, raw_val};
    }
    
    // Read tensor infos
    struct TempTensor {
        std::string name;
        std::vector<int64_t> shape;
        TensorType type;
        uint64_t offset;
    };
    std::vector<TempTensor> temp_tensors;
    
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name = read_gguf_str(file);
        uint32_t n_dims = read_u32(file);
        std::vector<int64_t> shape(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            shape[d] = read_u64(file);
        }
        uint32_t type = read_u32(file);
        uint64_t offset = read_u64(file);
        temp_tensors.push_back({name, shape, static_cast<TensorType>(type), offset});
    }
    
    // Determine data block start offset (aligned to alignment, default 32)
    uint32_t alignment = 32;
    uint32_t alignment_val = 0;
    if (get_metadata_uint32("general.alignment", alignment_val)) {
        alignment = alignment_val;
    }
    
    uint64_t current_pos = file.tellg();
    data_offset = (current_pos + alignment - 1) & ~(alignment - 1);
    
    // Sort temp tensors by offset to calculate sizes
    std::vector<size_t> indices(temp_tensors.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return temp_tensors[a].offset < temp_tensors[b].offset;
    });
    
    // Get file size
    file.seekg(0, std::ios::end);
    uint64_t file_size = file.tellg();
    
    // Build GGUFTensor structures
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        const auto& temp = temp_tensors[idx];
        
        uint64_t size_bytes = 0;
        if (i + 1 < indices.size()) {
            size_bytes = temp_tensors[indices[i+1]].offset - temp.offset;
        } else {
            size_bytes = file_size - data_offset - temp.offset;
        }
        
        GGUFTensor t;
        t.name = temp.name;
        t.shape = temp.shape;
        t.type = temp.type;
        t.offset = temp.offset;
        t.size_bytes = size_bytes;
        t.absolute_offset = data_offset + temp.offset;
        
        tensors[temp.name] = t;
        tensor_names.push_back(temp.name);
    }
    
    return true;
}

bool GGUFLoader::read_tensor_data(const GGUFTensor& tensor, void* buffer) {
    if (!file.is_open()) {
        std::cerr << "GGUFLoader::read_tensor_data error: File not open!" << std::endl;
        return false;
    }
    if (!file.good()) {
        file.clear(); // Clear stream error states
    }
    file.seekg(tensor.absolute_offset);
    if (!file.good()) {
        std::cerr << "GGUFLoader::read_tensor_data error: seekg to " << tensor.absolute_offset << " failed!" << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), tensor.size_bytes);
    if (!file.good()) {
        std::cerr << "GGUFLoader::read_tensor_data error: read of " << tensor.size_bytes << " bytes failed!" << std::endl;
        return false;
    }
    return true;
}

bool GGUFLoader::read_tensor_data(const std::string& name, void* buffer) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return false;
    }
    return read_tensor_data(it->second, buffer);
}

bool GGUFLoader::get_metadata_string(const std::string& key, std::string& val) const {
    auto it = metadata.find(key);
    if (it == metadata.end() || it->second.type != 8) return false;
    const auto& bytes = it->second.raw_data;
    if (bytes.size() < 8) return false;
    uint64_t len = *reinterpret_cast<const uint64_t*>(bytes.data());
    if (bytes.size() < 8 + len) return false;
    val = std::string(reinterpret_cast<const char*>(bytes.data() + 8), len);
    return true;
}

bool GGUFLoader::get_metadata_uint32(const std::string& key, uint32_t& val) const {
    auto it = metadata.find(key);
    if (it == metadata.end() || it->second.type != 4) return false;
    if (it->second.raw_data.size() < 4) return false;
    val = *reinterpret_cast<const uint32_t*>(it->second.raw_data.data());
    return true;
}

bool GGUFLoader::get_metadata_uint64(const std::string& key, uint64_t& val) const {
    auto it = metadata.find(key);
    if (it == metadata.end() || it->second.type != 10) return false;
    if (it->second.raw_data.size() < 8) return false;
    val = *reinterpret_cast<const uint64_t*>(it->second.raw_data.data());
    return true;
}

bool GGUFLoader::get_metadata_float(const std::string& key, float& val) const {
    auto it = metadata.find(key);
    if (it == metadata.end() || it->second.type != 6) return false;
    if (it->second.raw_data.size() < 4) return false;
    val = *reinterpret_cast<const float*>(it->second.raw_data.data());
    return true;
}

} // namespace turbo
