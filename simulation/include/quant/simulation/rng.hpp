#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace quant::simulation {

// SplitMix64 finalizer: decorrelates (seed, stream) pairs into independent generator states.
[[nodiscard]] inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/**
 * xoshiro256** (Blackman & Vigna) with a portable normal sampler. One instance per path (or per
 * bootstrap resample) keeps results reproducible independent of how work is distributed over threads.
 */
class PathRng {
public:
    PathRng(uint64_t seed, uint64_t stream) {
        uint64_t x = splitmix64(seed) ^ splitmix64(stream + 0x632BE59BD9B4E019ULL);
        for (auto& s : s_) {
            x = splitmix64(x);
            s = x;
        }
    }

    uint64_t next() noexcept {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform double in [0, 1) with 53 bits of randomness.
    double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    size_t index(size_t n) noexcept {
        const size_t i = static_cast<size_t>(uniform() * static_cast<double>(n));
        return i < n ? i : n - 1;
    }

    // Marsaglia polar method: exact N(0,1), identical output on every platform.
    double normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double m = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * m;
        has_spare_ = true;
        return u * m;
    }

private:
    static uint64_t rotl(uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4]{};
    double spare_{0.0};
    bool has_spare_{false};
};

} // namespace quant::simulation
