// Times the engine's compute-heavy workloads at several thread counts, for scripts/benchmark.py.
//
//   axiom_bench --data sample_data --threads 1,2,4 --repeats 5
//
// Prints CSV rows: workload,threads,seconds,check
// `seconds` is the fastest of the repeats; `check` is a summary statistic of the result (mean
// terminal wealth, or the bootstrap interval's lower bound) so the NumPy reference can confirm it
// computed the same thing.

#include "quant/analysis/significance.hpp"
#include "quant/data/csv_loader.hpp"
#include "quant/data/universe.hpp"
#include "quant/optimization/portfolio_stats.hpp"
#include "quant/simulation/monte_carlo.hpp"
#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kPaths = 50000;
constexpr size_t kHorizon = 252;
constexpr size_t kResamples = 10000;

std::vector<int> parse_threads(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const int t = std::stoi(item);
        if (t < 1) throw std::invalid_argument("thread counts must be positive");
        out.push_back(t);
    }
    if (out.empty()) throw std::invalid_argument("no thread counts given");
    return out;
}

// Runs `work` `repeats` times and returns the fastest wall-clock time with the last check value.
std::pair<double, double> time_best(int repeats, const std::function<double()>& work) {
    double best = std::numeric_limits<double>::infinity();
    double check = 0.0;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        check = work();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        best = std::min(best, seconds);
    }
    return {best, check};
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path data_dir = "sample_data";
        std::vector<int> threads{1};
        int repeats = 5;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--data") data_dir = value();
            else if (arg == "--threads") threads = parse_threads(value());
            else if (arg == "--repeats") repeats = std::max(1, std::stoi(value()));
            else throw std::invalid_argument("unknown option: " + arg);
        }

        auto raw = quant::data::CsvLoader::load_directory(data_dir);
        if (raw.size() < 2) throw std::runtime_error("need at least two CSV files in " + data_dir.string());
        std::vector<std::string> symbols;
        for (const auto& entry : raw) symbols.push_back(entry.first);
        std::sort(symbols.begin(), symbols.end());
        quant::data::MarketDataUniverse universe;
        for (const auto& s : symbols) universe.add_asset(s, std::move(raw[s]));
        universe.synchronize_timeline(true);

        const auto matrix = universe.get_aligned_returns_matrix();
        const auto rows = static_cast<Eigen::Index>(matrix.size());
        const auto cols = static_cast<Eigen::Index>(symbols.size());
        Eigen::MatrixXd returns(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) returns(r, c) = matrix[static_cast<size_t>(r)][static_cast<size_t>(c)];
        }
        // Column order is alphabetical; the first column drives the single-series workloads.
        std::vector<double> first(static_cast<size_t>(rows)), second(static_cast<size_t>(rows));
        for (Eigen::Index r = 0; r < rows; ++r) {
            first[static_cast<size_t>(r)] = returns(r, 0);
            second[static_cast<size_t>(r)] = returns(r, 1);
        }
        const Eigen::VectorXd mu = quant::optimization::PortfolioStats::compute_expected_returns(returns, 252.0);
        const Eigen::MatrixXd daily_cov = quant::optimization::PortfolioStats::compute_sample_covariance(returns, 1.0);
        const Eigen::VectorXd weights = Eigen::VectorXd::Constant(cols, 1.0 / static_cast<double>(cols));

        std::printf("workload,threads,seconds,check\n");
        for (int t : threads) {
#ifdef _OPENMP
            omp_set_num_threads(t);
#endif
            quant::simulation::MonteCarloConfig mc;
            mc.num_simulations = kPaths;
            mc.horizon_days = kHorizon;
            mc.num_threads = static_cast<size_t>(t);

            mc.use_bootstrap = true;
            const quant::simulation::MonteCarloEngine bootstrap_engine(mc);
            auto [boot_s, boot_check] = time_best(repeats, [&] {
                return bootstrap_engine.run_simulation(first).mean_terminal_wealth;
            });
            std::printf("mc_bootstrap,%d,%.6f,%.6f\n", t, boot_s, boot_check);

            mc.use_bootstrap = false;
            const quant::simulation::MonteCarloEngine gbm_engine(mc);
            auto [gbm_s, gbm_check] = time_best(repeats, [&] {
                return gbm_engine.run_gbm_portfolio(mu, daily_cov, weights).mean_terminal_wealth;
            });
            std::printf("mc_gbm_portfolio,%d,%.6f,%.6f\n", t, gbm_s, gbm_check);

            quant::analysis::BootstrapConfig bc;
            bc.resamples = kResamples;
            auto [sb_s, sb_check] = time_best(repeats, [&] {
                return quant::analysis::bootstrap_sharpe(first, second, bc).strategy.lower;
            });
            std::printf("block_bootstrap_sharpe,%d,%.6f,%.6f\n", t, sb_s, sb_check);
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
