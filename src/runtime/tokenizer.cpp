#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>

namespace turbo {

Tokenizer::Tokenizer(const Model& model) {
    vocab = model.vocab_tokens;
    for (size_t i = 0; i < vocab.size(); ++i) {
        token_to_id[vocab[i]] = static_cast<int>(i);
    }

    // Attempt to auto-detect special tokens
    auto find_special = [&](const std::string& name) -> int {
        auto it = token_to_id.find(name);
        if (it != token_to_id.end()) return it->second;
        return -1;
    };

    bos_id = find_special("<|im_start|>");
    if (bos_id == -1) bos_id = find_special("<s>");
    
    eos_id = find_special("<|im_end|>");
    if (eos_id == -1) eos_id = find_special("<|endoftext|>");
    if (eos_id == -1) eos_id = find_special("</s>");

    pad_id = find_special("<|endoftext|>");
    if (pad_id == -1) pad_id = find_special("<pad>");
}

Tokenizer::~Tokenizer() {}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.length()) {
        // Find the longest matching token from the current position
        int matched_id = -1;
        size_t matched_len = 0;

        for (size_t len = 1; len <= text.length() - i; ++len) {
            std::string sub = text.substr(i, len);
            auto it = token_to_id.find(sub);
            if (it != token_to_id.end()) {
                matched_id = it->second;
                matched_len = len;
            }
        }

        if (matched_id != -1) {
            ids.push_back(matched_id);
            i += matched_len;
        } else {
            // Fallback for single byte/unk characters
            std::string fallback_char(1, text[i]);
            auto it = token_to_id.find(fallback_char);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                // If not found, use a placeholder byte/UNK ID
                ids.push_back(0); // fallback to ID 0
            }
            i += 1;
        }
    }
    return ids;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < static_cast<int>(vocab.size())) {
        std::string token = vocab[token_id];
        // Clean up GGUF byte representation like <0x0A> or special escapes
        if (token.length() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
            char hex[3] = { token[3], token[4], '\0' };
            char c = static_cast<char>(std::strtol(hex, nullptr, 16));
            return std::string(1, c);
        }
        // Clean up standard GGUF space characters (usually byte-level BPE structures)
        // or standard GGUF formatting
        return token;
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string text = "";
    for (int id : token_ids) {
        text += decode(id);
    }
    return text;
}

} // namespace turbo
