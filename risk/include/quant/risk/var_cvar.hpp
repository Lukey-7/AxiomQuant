#pragma once

#include <vector>

namespace quant::risk {

struct VarCvarResult {
    double var{0.0};    // Value at Risk (expressed as a positive loss percentage, e.g. 0.02 = 2% loss)
    double cvar{0.0};   // Conditional Value at Risk / Expected Shortfall (positive loss percentage)
};

class ValueAtRisk {
public:
    /**
     * @brief Computes Historical (empirical non-parametric) VaR and CVaR.
     * @param returns Daily return series.
     * @param confidence_level Confidence level (e.g. 0.95 for 95% confidence).
     * @return VarCvarResult with positive loss percentages.
     */
    [[nodiscard]] static VarCvarResult historical(const std::vector<double>& returns,
                                                  double confidence_level = 0.95);

    /**
     * @brief Computes Parametric (Gaussian) VaR and CVaR.
     * @param returns Daily return series.
     * @param confidence_level Confidence level (e.g. 0.95).
     */
    [[nodiscard]] static VarCvarResult parametric(const std::vector<double>& returns,
                                                  double confidence_level = 0.95);

    /**
     * @brief Computes Cornish-Fisher expansion VaR and CVaR (adjusting for skewness and kurtosis).
     * @param returns Daily return series.
     * @param confidence_level Confidence level (e.g. 0.95).
     */
    [[nodiscard]] static VarCvarResult cornish_fisher(const std::vector<double>& returns,
                                                      double confidence_level = 0.95);

    /**
     * @brief Inverse standard normal cumulative distribution function (quantile function / probit).
     */
    [[nodiscard]] static double standard_normal_quantile(double p);

    /**
     * @brief Standard normal probability density function: phi(x) = (1/sqrt(2pi)) * exp(-x^2 / 2).
     */
    [[nodiscard]] static double standard_normal_pdf(double x);
};

}   // namespace quant::risk
