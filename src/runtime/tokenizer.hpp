#pragma once

#include "../model/model.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace turbo {

class Tokenizer {
private:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;

public:
    Tokenizer(const Model& model);
    ~Tokenizer();

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    int get_bos_id() const { return bos_id; }
    int get_eos_id() const { return eos_id; }
    int get_pad_id() const { return pad_id; }
};

} // namespace turbo
