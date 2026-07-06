#include "sampling.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>

namespace turbo {

int Sampler::sample(const float* logits, int vocab_size, float temperature, float top_p) {
    if (temperature <= 0.0f) {
        // Argmax (Greedy)
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // 1. Calculate Softmax probabilities with temperature scaling
    std::vector<float> probs(vocab_size);
    float max_val = *std::max_element(logits, logits + vocab_size);
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // 2. Apply Top-P (Nucleus) Sampling
    if (top_p < 1.0f) {
        struct Candidate {
            int id;
            float prob;
        };
        std::vector<Candidate> candidates(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            candidates[i] = {i, probs[i]};
        }

        // Sort by probability descending
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.prob > b.prob;
        });

        // Find Top-P threshold
        float cum_sum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cum_sum += candidates[i].prob;
            if (cum_sum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Normalize truncated candidates
        float sum_truncated = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            sum_truncated += candidates[i].prob;
        }
        for (int i = 0; i < cutoff; ++i) {
            candidates[i].prob /= sum_truncated;
        }

        // Sample
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(gen);
        
        float current_sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            current_sum += candidates[i].prob;
            if (r <= current_sum) {
                return candidates[i].id;
            }
        }
        return candidates[cutoff - 1].id;
    } else {
        // Sample directly from full distribution
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(gen);
        
        float current_sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            current_sum += probs[i];
            if (r <= current_sum) {
                return i;
            }
        }
        return vocab_size - 1;
    }
}

} // namespace turbo
