#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace quant::data {

/**
 * @brief Which symbols belonged to an index on a given date.
 *
 * Backtests that rank a fixed list of today's symbols are survivorship-biased: the list is the set
 * of companies that survived and grew, chosen with knowledge the strategy could not have had. A
 * membership calendar removes that bias by answering "was this symbol in the index on this date?"
 * from a point-in-time record, so a cross-sectional strategy only ever chooses among the names it
 * could have chosen at the time.
 *
 * The file is CSV with a header and one row per membership spell:
 *
 *     ticker,start_date,end_date
 *     AAPL,1982-11-30,
 *     LEH,1994-09-08,2008-09-15
 *
 * Dates are inclusive and lexicographically comparable (YYYY-MM-DD); an empty `end_date` means the
 * symbol is still a member. A symbol may appear several times if it left and rejoined.
 */
class MembershipCalendar {
public:
    struct Spell {
        std::string start_date;   // inclusive, may be empty for "always was"
        std::string end_date;     // inclusive, empty for "still a member"
    };

    /// An empty calendar treats every symbol as a member on every date, so callers can always hold one.
    MembershipCalendar() = default;

    /// @throws std::runtime_error if the file cannot be opened or has no usable rows.
    [[nodiscard]] static MembershipCalendar load(const std::filesystem::path& path);

    void add_spell(const std::string& ticker, const std::string& start_date, const std::string& end_date);

    [[nodiscard]] bool empty() const noexcept { return spells_.empty(); }
    [[nodiscard]] size_t ticker_count() const noexcept { return spells_.size(); }

    /// True when `ticker` was a member on `date` (or when the calendar is empty).
    [[nodiscard]] bool is_member(const std::string& ticker, const std::string& date) const;

    /// Symbols that were members on `date`, sorted.
    [[nodiscard]] std::vector<std::string> members_on(const std::string& date) const;

private:
    std::unordered_map<std::string, std::vector<Spell>> spells_;
};

}   // namespace quant::data
