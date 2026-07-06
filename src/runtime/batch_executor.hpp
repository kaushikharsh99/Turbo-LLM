#pragma once

#include "../model/model.hpp"
#include "scheduler.hpp"
#include "executor.hpp"
#include "tokenizer.hpp"
#include "../config.hpp"
#include <memory>
#include <string>
#include <vector>

namespace turbo {

class BatchExecutor {
private:
    const Model& model;
    std::shared_ptr<MemoryManager> memory_manager;
    std::shared_ptr<Scheduler> scheduler;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<Tokenizer> tokenizer;
    const Config& config;

public:
    BatchExecutor(
        const Model& m, 
        std::shared_ptr<MemoryManager> mem_mgr, 
        std::shared_ptr<Scheduler> sched,
        const Config& cfg
    );
    ~BatchExecutor();

    void run(const std::vector<std::string>& prompts);
};

} // namespace turbo
