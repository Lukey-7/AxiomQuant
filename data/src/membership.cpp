#include "quant/data/membership.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace quant::data {

namespace {

std::string trim(std::string text) {
    const auto not_space = [](unsigned char c) { return c != ' ' && c != '\t' && c != '\r' && c != '\n'; };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
        fields.push_back(trim(field));
    return fields;
}

int column_of(const std::vector<std::string>& header, const std::vector<std::string>& names) {
    for (size_t i = 0; i < header.size(); ++i) {
        std::string lowered = header[i];
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (std::find(names.begin(), names.end(), lowered) != names.end()) return static_cast<int>(i);
    }
    return -1;
}

}   // namespace

MembershipCalendar MembershipCalendar::load(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open membership file: " + path.string());
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Membership file is empty: " + path.string());
    }
    const auto header = split(line);
    const int ticker_idx = column_of(header, {"ticker", "symbol", "name"});
    const int start_idx = column_of(header, {"start_date", "start", "from", "added"});
    const int end_idx = column_of(header, {"end_date", "end", "to", "removed"});
    if (ticker_idx < 0 || start_idx < 0) {
        throw std::runtime_error("Membership file needs 'ticker' and 'start_date' columns: " + path.string());
    }

    MembershipCalendar calendar;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split(line);
        const auto widest = static_cast<size_t>(std::max(ticker_idx, start_idx));
        if (fields.size() <= widest) continue;
        const std::string& ticker = fields[static_cast<size_t>(ticker_idx)];
        if (ticker.empty()) continue;
        const std::string end_date = (end_idx >= 0 && fields.size() > static_cast<size_t>(end_idx))
                                         ? fields[static_cast<size_t>(end_idx)]
                                         : std::string{};
        calendar.add_spell(ticker, fields[static_cast<size_t>(start_idx)], end_date);
    }
    if (calendar.empty()) {
        throw std::runtime_error("Membership file has no usable rows: " + path.string());
    }
    return calendar;
}

void MembershipCalendar::add_spell(const std::string& ticker,
                                   const std::string& start_date,
                                   const std::string& end_date) {
    std::string upper = ticker;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    spells_[upper].push_back({start_date, end_date});
}

bool MembershipCalendar::is_member(const std::string& ticker, const std::string& date) const {
    if (spells_.empty()) return true;   // no calendar loaded: every symbol counts
    std::string upper = ticker;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    const auto it = spells_.find(upper);
    if (it == spells_.end()) return false;
    for (const auto& spell : it->second) {
        // Dates are ISO-8601, so string comparison is date comparison. Both ends are inclusive.
        if (!spell.start_date.empty() && date < spell.start_date) continue;
        if (!spell.end_date.empty() && date > spell.end_date) continue;
        return true;
    }
    return false;
}

std::vector<std::string> MembershipCalendar::members_on(const std::string& date) const {
    std::vector<std::string> members;
    members.reserve(spells_.size());
    for (const auto& [ticker, spells] : spells_) {
        if (is_member(ticker, date)) members.push_back(ticker);
    }
    std::sort(members.begin(), members.end());
    return members;
}

}   // namespace quant::data
