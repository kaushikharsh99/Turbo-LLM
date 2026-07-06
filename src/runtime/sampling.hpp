#pragma once

#include <vector>

namespace turbo {

class Sampler {
public:
    static int sample(
        const float* logits, 
        int vocab_size, 
        float temperature, 
        float top_p
    );
};

} // namespace turbo
