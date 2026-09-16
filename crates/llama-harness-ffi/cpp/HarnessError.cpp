// HarnessError.cpp -- encode() helper. Self-contained; no llama.cpp
// dependency.

#include "harness.h"

#include <cstdio>

namespace harness {

std::string HarnessError::encode() const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d:", static_cast<int>(kind_));
    return std::string(buf) + std::string(msg_);
}

}  // namespace harness
