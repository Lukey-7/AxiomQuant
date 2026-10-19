// Compares a program output against its checked-in golden copy.
//
//   golden_diff [--ignore <substring>]... <expected> <actual>
//   golden_diff [--ignore <substring>]... --write <output> <golden copy>   (used by update_golden)
//
// Lines are split into numbers and text. Text must match exactly, whitespace aside (column padding
// shifts by one when a value gains a minus sign). Integers must match exactly; a decimal may differ by
// at most one unit in its last printed digit, so a libm that rounds the final bit of exp()
// differently on another platform passes, while a change to a formula, a default or a fill rule
// moves far more than one figure by far more than that. Lines containing an
// --ignore substring (timings, thread counts, paths) are dropped from both files first.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Token {
    bool is_number{false};
    std::string text;
    double value{0.0};
    int decimals{0};
};

struct Line {
    size_t number{0};   // 1-based line in the original file
    std::string text;
};

std::vector<Line> read_lines(const std::string& path, const std::vector<std::string>& ignore) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "golden_diff: cannot open " << path << "\n";
        std::exit(2);
    }
    std::vector<Line> lines;
    std::string text;
    size_t number = 0;
    while (std::getline(in, text)) {
        ++number;
        if (!text.empty() && text.back() == '\r') text.pop_back();
        const bool skipped = std::any_of(ignore.begin(), ignore.end(), [&text](const std::string& s) {
            return text.find(s) != std::string::npos;
        });
        const bool blank =
            std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c); });
        if (!skipped && !blank) lines.push_back({number, text});
    }
    return lines;
}

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

std::vector<Token> tokenize(const std::string& line) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < line.size()) {
        const char c = line[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        // A sign belongs to the number only when it cannot be a separator, so 2020-01-02 stays a date.
        const bool prev_joins =
            i > 0 && (std::isalnum(static_cast<unsigned char>(line[i - 1])) || line[i - 1] == '.');
        const bool signed_start =
            (c == '-' || c == '+') && !prev_joins && i + 1 < line.size() && is_digit(line[i + 1]);
        if (is_digit(c) || signed_start) {
            size_t j = i + 1;
            while (j < line.size() && is_digit(line[j]))
                ++j;
            int decimals = 0;
            if (j + 1 < line.size() && line[j] == '.' && is_digit(line[j + 1])) {
                ++j;
                while (j < line.size() && is_digit(line[j])) {
                    ++j;
                    ++decimals;
                }
            }
            Token token;
            token.is_number = true;
            token.text = line.substr(i, j - i);
            token.value = std::strtod(token.text.c_str(), nullptr);
            token.decimals = decimals;
            tokens.push_back(token);
            i = j;
            continue;
        }
        Token token;
        token.text = std::string(1, c);
        tokens.push_back(token);
        ++i;
    }
    return tokens;
}

/// One unit in the last printed decimal, plus a hair for the binary representation. Integers (fill
/// counts, bar counts, dates) must match exactly.
double tolerance(const Token& expected) {
    if (expected.decimals == 0) return 0.0;
    return std::pow(10.0, -expected.decimals) * 1.000001;
}

bool same_line(const std::string& expected, const std::string& actual, std::string& why) {
    const auto want = tokenize(expected);
    const auto got = tokenize(actual);
    if (want.size() != got.size()) {
        why = "different number of fields";
        return false;
    }
    for (size_t k = 0; k < want.size(); ++k) {
        const auto& w = want[k];
        const auto& g = got[k];
        if (w.is_number != g.is_number) {
            why = "'" + w.text + "' became '" + g.text + "'";
            return false;
        }
        if (!w.is_number) {
            if (w.text != g.text) {
                why = "'" + w.text + "' became '" + g.text + "'";
                return false;
            }
            continue;
        }
        if (w.decimals != g.decimals || std::abs(w.value - g.value) > tolerance(w)) {
            why = w.text + " became " + g.text;
            return false;
        }
    }
    return true;
}

/// Copies `from` to `to` without the ignored lines, so a golden copy never records a machine's paths
/// or timings.
int write_filtered(const std::string& from, const std::string& to, const std::vector<std::string>& ignore) {
    std::ifstream in(from, std::ios::binary);
    std::ofstream out(to, std::ios::binary);
    if (!in || !out) {
        std::cerr << "golden_diff: cannot copy " << from << " to " << to << "\n";
        return 2;
    }
    std::string text;
    while (std::getline(in, text)) {
        if (!text.empty() && text.back() == '\r') text.pop_back();
        const bool skipped = std::any_of(ignore.begin(), ignore.end(), [&text](const std::string& s) {
            return text.find(s) != std::string::npos;
        });
        if (!skipped) out << text << '\n';
    }
    return 0;
}
}   // namespace

int main(int argc, char* argv[]) {
    std::vector<std::string> ignore;
    std::vector<std::string> files;
    bool write = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ignore" && i + 1 < argc) ignore.emplace_back(argv[++i]);
        else if (arg == "--write") write = true;
        else files.push_back(arg);
    }
    if (files.size() != 2) {
        std::cerr << "usage: golden_diff [--ignore <substring>]... <expected> <actual>\n"
                     "       golden_diff [--ignore <substring>]... --write <output> <golden copy>\n";
        return 2;
    }
    if (write) return write_filtered(files[0], files[1], ignore);

    const auto expected = read_lines(files[0], ignore);
    const auto actual = read_lines(files[1], ignore);

    size_t mismatches = 0;
    const size_t common = std::min(expected.size(), actual.size());
    for (size_t k = 0; k < common; ++k) {
        std::string why;
        if (same_line(expected[k].text, actual[k].text, why)) continue;
        if (++mismatches <= 20) {
            std::cerr << files[1] << ":" << actual[k].number << ": " << why << "\n"
                      << "  expected: " << expected[k].text << "\n"
                      << "  actual:   " << actual[k].text << "\n";
        }
    }
    if (expected.size() != actual.size()) {
        ++mismatches;
        std::cerr << files[1] << ": " << actual.size() << " compared lines, golden copy has "
                  << expected.size() << "\n";
    }

    if (mismatches > 0) {
        std::cerr << "golden_diff: " << files[1] << " differs from " << files[0] << " (" << mismatches
                  << " mismatched lines). If the change is intended, regenerate with: "
                     "cmake --build <build> --target update_golden\n";
        return 1;
    }
    std::cout << "golden_diff: " << files[1] << " matches (" << common << " lines)\n";
    return 0;
}
