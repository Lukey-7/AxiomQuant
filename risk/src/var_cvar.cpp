#include "quant/risk/var_cvar.hpp"
#include "quant/risk/metrics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <numbers>

namespace quant::risk {

// High-precision Acklam inverse normal CDF approximation
double ValueAtRisk::standard_normal_quantile(double p) {
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return 10.0;

    // Coefficients in rational approximations
    constexpr double a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    constexpr double b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    constexpr double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    constexpr double d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    double q = 0.0;
    if (p < p_low) {
        // Rational approximation for lower region
        double r = std::sqrt(-2.0 * std::log(p));
        q = (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) /
            ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1.0);
    } else if (p <= p_high) {
        // Rational approximation for central region
        double r = p - 0.5;
        double s = r * r;
        q = (((((a[0]*s + a[1])*s + a[2])*s + a[3])*s + a[4])*s + a[5])*r /
            (((((b[0]*s + b[1])*s + b[2])*s + b[3])*s + b[4])*s + 1.0);
    } else {
        // Rational approximation for upper region
        double r = std::sqrt(-2.0 * std::log(1.0 - p));
        q = -(((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) /
             ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1.0);
    }

    return q;
}

double ValueAtRisk::standard_normal_pdf(double x) {
    constexpr double inv_sqrt_2pi = 0.398942280401432677939946059934; // 1 / sqrt(2*pi)
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

VarCvarResult ValueAtRisk::historical(const std::vector<double>& returns, double confidence_level) {
    VarCvarResult result;
    if (returns.empty()) return result;

    std::vector<double> sorted = returns;
    std::sort(sorted.begin(), sorted.end());

    double alpha = 1.0 - confidence_level;
    size_t index = static_cast<size_t>(std::floor(alpha * static_cast<double>(sorted.size())));
    if (index >= sorted.size()) index = sorted.size() - 1;

    // VaR is positive loss
    result.var = -sorted[index];

    // CVaR is the mean loss of all observations <= VaR threshold
    double sum_tail = 0.0;
    for (size_t i = 0; i <= index; ++i) {
        sum_tail += sorted[i];
    }
    result.cvar = -(sum_tail / static_cast<double>(index + 1));

    return result;
}

VarCvarResult ValueAtRisk::parametric(const std::vector<double>& returns, double confidence_level) {
    VarCvarResult result;
    if (returns.size() < 2) return result;

    double mu = RiskMetrics::mean(returns);
    double sigma = RiskMetrics::standard_deviation(returns);

    // Z value for upper tail of losses = quantile at confidence_level
    double z = standard_normal_quantile(confidence_level);
    double alpha = 1.0 - confidence_level;

    result.var = -mu + z * sigma;
    // Expected shortfall for normal distribution: mu + sigma * (phi(z) / (1 - alpha))
    double pdf_z = standard_normal_pdf(z);
    result.cvar = -mu + sigma * (pdf_z / alpha);

    return result;
}

VarCvarResult ValueAtRisk::cornish_fisher(const std::vector<double>& returns, double confidence_level) {
    VarCvarResult result;
    if (returns.size() < 4) return parametric(returns, confidence_level);

    double mu = RiskMetrics::mean(returns);
    double sigma = RiskMetrics::standard_deviation(returns);
    double s = RiskMetrics::skewness(returns);
    double k = RiskMetrics::excess_kurtosis(returns);

    double z = standard_normal_quantile(confidence_level);

    // Cornish-Fisher expansion for quantile
    double z_cf = z + (s / 6.0) * (z * z - 1.0)
                    + (k / 24.0) * (z * z * z - 3.0 * z)
                    - (s * s / 36.0) * (2.0 * z * z * z - 5.0 * z);

    result.var = -mu + z_cf * sigma;

    // Approximate CVaR with CF adjusted tail scaling
    double alpha = 1.0 - confidence_level;
    double pdf_z = standard_normal_pdf(z_cf);
    result.cvar = -mu + sigma * (pdf_z / alpha);

    return result;
}

} // namespace quant::risk
