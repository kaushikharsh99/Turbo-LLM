#pragma once

#include "../gguf/loader.hpp"
#include "model.hpp"
#include <memory>

namespace turbo {

class Parser {
public:
    static std::unique_ptr<Model> parse(const GGUFLoader& loader);
};

} // namespace turbo
