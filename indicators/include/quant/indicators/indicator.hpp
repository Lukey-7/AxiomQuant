#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cstddef>
#include <string>

namespace quant::indicators {

inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

[[nodiscard]] inline bool is_nan(double val) noexcept {
    return std::isnan(val);
}

[[nodiscard]] inline bool is_valid(double val) noexcept {
    return !std::isnan(val) && !std::isinf(val);
}

}   // namespace quant::indicators
