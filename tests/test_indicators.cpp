#include "test_harness.hpp"
#include "quant/indicators/sma.hpp"
#include "quant/indicators/ema.hpp"
#include "quant/indicators/rsi.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/rolling_volatility.hpp"
#include "quant/indicators/atr.hpp"

TEST_CASE(TestIndicators_SMA) {
    std::vector<double> prices = {10.0, 20.0, 30.0, 40.0, 50.0};
    auto sma3 = quant::indicators::SMA::calculate(prices, 3);

    EXPECT_EQ(sma3.size(), 5);
    EXPECT_TRUE(quant::indicators::is_nan(sma3[0]));
    EXPECT_TRUE(quant::indicators::is_nan(sma3[1]));
    EXPECT_NEAR(sma3[2], 20.0, 1e-6);   // (10 + 20 + 30) / 3 = 20
    EXPECT_NEAR(sma3[3], 30.0, 1e-6);   // (20 + 30 + 40) / 3 = 30
    EXPECT_NEAR(sma3[4], 40.0, 1e-6);   // (30 + 40 + 50) / 3 = 40
}

TEST_CASE(TestIndicators_EMA) {
    std::vector<double> prices = {2.0, 4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0};
    auto ema3 = quant::indicators::EMA::calculate(prices, 3);

    EXPECT_EQ(ema3.size(), prices.size());
    EXPECT_TRUE(quant::indicators::is_nan(ema3[0]));
    EXPECT_TRUE(quant::indicators::is_nan(ema3[1]));
    // Seed at index 2 is SMA of first 3: (2 + 4 + 6)/3 = 4.0
    EXPECT_NEAR(ema3[2], 4.0, 1e-6);

    // alpha = 2 / (3 + 1) = 0.5
    // at index 3: 0.5 * 8.0 + 0.5 * 4.0 = 6.0
    EXPECT_NEAR(ema3[3], 6.0, 1e-6);
    // at index 4: 0.5 * 12.0 + 0.5 * 6.0 = 9.0
    EXPECT_NEAR(ema3[4], 9.0, 1e-6);
}

TEST_CASE(TestIndicators_RSI_Monotonic) {
    // Strictly increasing prices -> RSI should be 100.0
    std::vector<double> rising(20);
    for (size_t i = 0; i < 20; ++i)
        rising[i] = 100.0 + static_cast<double>(i) * 5.0;

    auto rsi_up = quant::indicators::RSI::calculate(rising, 14);
    EXPECT_NEAR(rsi_up.back(), 100.0, 1e-4);

    // Strictly decreasing prices -> RSI should be 0.0
    std::vector<double> falling(20);
    for (size_t i = 0; i < 20; ++i)
        falling[i] = 200.0 - static_cast<double>(i) * 5.0;

    auto rsi_down = quant::indicators::RSI::calculate(falling, 14);
    EXPECT_NEAR(rsi_down.back(), 0.0, 1e-4);
}

TEST_CASE(TestIndicators_MACD) {
    std::vector<double> prices(50);
    for (size_t i = 0; i < 50; ++i) {
        prices[i] = 100.0 + std::sin(static_cast<double>(i) * 0.2) * 10.0;
    }

    auto macd_res = quant::indicators::MACD::calculate(prices, 12, 26, 9);
    EXPECT_EQ(macd_res.macd_line.size(), 50);
    EXPECT_EQ(macd_res.signal_line.size(), 50);
    EXPECT_EQ(macd_res.histogram.size(), 50);

    // Histogram must equal macd_line - signal_line for valid entries
    for (size_t i = 35; i < 50; ++i) {
        EXPECT_TRUE(quant::indicators::is_valid(macd_res.histogram[i]));
        EXPECT_NEAR(macd_res.histogram[i], macd_res.macd_line[i] - macd_res.signal_line[i], 1e-7);
    }
}

TEST_CASE(TestIndicators_BollingerBands) {
    std::vector<double> prices(30, 50.0);
    // Add small oscillation
    for (size_t i = 0; i < 30; ++i) {
        prices[i] += (i % 2 == 0 ? 2.0 : -2.0);
    }

    auto bb = quant::indicators::BollingerBands::calculate(prices, 10, 2.0);
    EXPECT_EQ(bb.upper.size(), 30);
    EXPECT_EQ(bb.lower.size(), 30);
    EXPECT_EQ(bb.middle.size(), 30);

    for (size_t i = 15; i < 30; ++i) {
        EXPECT_TRUE(bb.upper[i] > bb.middle[i]);
        EXPECT_TRUE(bb.lower[i] < bb.middle[i]);
        EXPECT_NEAR(bb.middle[i] - bb.lower[i], bb.upper[i] - bb.middle[i], 1e-6);
    }
}

TEST_CASE(TestIndicators_RollingVolatility) {
    std::vector<double> prices = {100.0, 102.0, 101.0, 103.0, 102.5, 104.0};
    auto vol = quant::indicators::RollingVolatility::calculate(prices, 3, 252.0);
    EXPECT_EQ(vol.size(), prices.size());
    EXPECT_TRUE(quant::indicators::is_valid(vol.back()));
    EXPECT_TRUE(vol.back() > 0.0);
}

TEST_CASE(TestIndicators_ATR) {
    std::vector<double> high = {10.0, 12.0, 15.0, 14.0, 16.0};
    std::vector<double> low = {8.0, 9.0, 11.0, 10.0, 12.0};
    std::vector<double> close = {9.0, 11.0, 13.0, 12.0, 15.0};

    auto atr = quant::indicators::ATR::calculate(high, low, close, 3);
    EXPECT_EQ(atr.size(), 5);
    EXPECT_TRUE(quant::indicators::is_valid(atr[2]));
    EXPECT_TRUE(atr.back() > 0.0);
}
