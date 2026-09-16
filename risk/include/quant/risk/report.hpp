#pragma once

#include "quant/risk/metrics.hpp"
#include "quant/backtest/engine.hpp"
#include <string>
#include <vector>

namespace quant::risk {

class RiskReport {
public:
    /**
     * @brief Evaluates complete performance summary from backtest result.
     */
    [[nodiscard]] static PerformanceSummary evaluate(const backtest::BacktestResult& result,
                                                     double risk_free_rate = 0.02);

    /**
     * @brief Generates formatted text report with tables and risk indicators.
     */
    [[nodiscard]] static std::string generate_text_report(const backtest::BacktestResult& result,
                                                          const PerformanceSummary& summary);

    /**
     * @brief Generates an ASCII line chart of the portfolio equity curve.
     * @param equity_points Vector of equity values.
     * @param width Character width of the chart.
     * @param height Line height of the chart.
     */
    [[nodiscard]] static std::string render_ascii_chart(const std::vector<double>& equity_points,
                                                        size_t width = 60,
                                                        size_t height = 15);
};

}   // namespace quant::risk
