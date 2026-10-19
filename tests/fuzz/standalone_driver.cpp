// Runs a libFuzzer-style target without libFuzzer, so every compiler in CI exercises the fuzz target:
// each corpus file is replayed as-is, then mutated a fixed number of times with a seeded generator.
//
//   fuzz_csv_loader_replay [--mutations N] [--seed S] <file-or-directory>...

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace {

namespace fs = std::filesystem;

std::vector<uint8_t> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void run_one(const std::vector<uint8_t>& bytes) {
    static const uint8_t empty_input = 0;   // libFuzzer never passes a null pointer, even for size 0
    LLVMFuzzerTestOneInput(bytes.empty() ? &empty_input : bytes.data(), bytes.size());
}

/// Byte-level mutations in the spirit of libFuzzer's defaults, plus CSV-aware ones (separators,
/// numbers that parse to inf/nan/negative), which reach the row parser far more often.
std::vector<uint8_t> mutate(std::vector<uint8_t> bytes, std::mt19937_64& rng) {
    static const std::vector<std::string> tokens{
        ",",    "\n", "\"", "-",          "inf",  "nan",  "1e308", "-1",  "0",     "1e-320",
        "\r\n", "#",  " ",  "9999-99-99", "Date", "Open", "High",  "Low", "Close", "Volume"};
    const int steps = 1 + static_cast<int>(rng() % 4);
    for (int s = 0; s < steps; ++s) {
        const size_t pos = bytes.empty() ? 0 : static_cast<size_t>(rng() % (bytes.size() + 1));
        switch (rng() % 5) {
            case 0:   // flip a byte
                if (!bytes.empty()) bytes[pos % bytes.size()] = static_cast<uint8_t>(rng());
                break;
            case 1:   // delete a run
                if (!bytes.empty()) {
                    const size_t start = pos % bytes.size();
                    const size_t len = std::min<size_t>(bytes.size() - start, 1 + rng() % 16);
                    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(start),
                                bytes.begin() + static_cast<std::ptrdiff_t>(start + len));
                }
                break;
            case 2: {   // insert a CSV-meaningful token
                const auto& token = tokens[rng() % tokens.size()];
                bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(pos), token.begin(), token.end());
                break;
            }
            case 3:   // duplicate a run (repeated rows, long lines)
                if (!bytes.empty()) {
                    const size_t start = pos % bytes.size();
                    const size_t len = std::min<size_t>(bytes.size() - start, 1 + rng() % 64);
                    const std::vector<uint8_t> chunk(
                        bytes.begin() + static_cast<std::ptrdiff_t>(start),
                        bytes.begin() + static_cast<std::ptrdiff_t>(start + len));
                    bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(pos), chunk.begin(),
                                 chunk.end());
                }
                break;
            default:   // truncate
                bytes.resize(pos);
                break;
        }
    }
    return bytes;
}

}   // namespace

int main(int argc, char* argv[]) {
    size_t mutations = 0;
    uint64_t seed = 1;
    std::vector<fs::path> inputs;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--mutations" && i + 1 < argc) mutations = std::stoull(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else inputs.emplace_back(arg);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bad argument: %s\n", e.what());
        return 2;
    }

    std::vector<fs::path> files;
    for (const auto& input : inputs) {
        if (fs::is_directory(input)) {
            for (const auto& entry : fs::directory_iterator(input)) {
                if (entry.is_regular_file()) files.push_back(entry.path());
            }
        } else if (fs::is_regular_file(input)) {
            files.push_back(input);
        } else {
            std::fprintf(stderr, "no such file or directory: %s\n", input.string().c_str());
            return 2;
        }
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: %s [--mutations N] [--seed S] <file-or-directory>...\n", argv[0]);
        return 2;
    }
    std::sort(files.begin(), files.end());   // directory order differs by platform

    std::mt19937_64 rng(seed);
    size_t executions = 0;
    for (const auto& file : files) {
        const auto bytes = read_file(file);
        run_one(bytes);
        ++executions;
        for (size_t m = 0; m < mutations; ++m) {
            run_one(mutate(bytes, rng));
            ++executions;
        }
    }
    std::printf("%zu inputs from %zu corpus files ran without violating an invariant\n", executions,
                files.size());
    return 0;
}
