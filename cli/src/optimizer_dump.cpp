// Reads a portfolio problem from a text file and prints every optimizer's weights, one line per result,
// so scripts/check_optimizer.py can compare them with an independent reference solver.
//
// Input (whitespace separated):
//   n
//   mu_1 ... mu_n
//   Sigma (n rows of n values)
//   risk_free min_weight max_weight
//   G gamma_1 ... gamma_G
//   K target_1 ... target_K
//   P then P projection vectors of n values each
//
// Output lines: "<name> w_1 ... w_n", values printed with 17 significant digits.

#include "quant/optimization/constrained_qp.hpp"
#include "quant/optimization/unconstrained.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace opt = quant::optimization;

namespace {

void print(const std::string& name, const Eigen::VectorXd& w) {
    std::printf("%s", name.c_str());
    for (Eigen::Index i = 0; i < w.size(); ++i)
        std::printf(" %.17g", w(i));
    std::printf("\n");
}

double read_double(std::istream& in) {
    double v = 0.0;
    if (!(in >> v)) throw std::runtime_error("unexpected end of input");
    return v;
}

size_t read_count(std::istream& in) {
    long long v = 0;
    if (!(in >> v) || v < 0) throw std::runtime_error("expected a non-negative count");
    return static_cast<size_t>(v);
}

Eigen::VectorXd read_vector(std::istream& in, size_t n) {
    Eigen::VectorXd v(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < v.size(); ++i)
        v(i) = read_double(in);
    return v;
}

}   // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: axiom_optimizer_dump <problem.txt>\n";
        return 2;
    }
    try {
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error(std::string("cannot open ") + argv[1]);

        const size_t n = read_count(in);
        const Eigen::VectorXd mu = read_vector(in, n);
        Eigen::MatrixXd sigma(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        for (Eigen::Index r = 0; r < sigma.rows(); ++r)
            sigma.row(r) = read_vector(in, n).transpose();
        const double rf = read_double(in);
        const double min_w = read_double(in);
        const double max_w = read_double(in);

        std::vector<double> gammas(read_count(in));
        for (double& g : gammas)
            g = read_double(in);
        std::vector<double> targets(read_count(in));
        for (double& t : targets)
            t = read_double(in);

        print("gmv_unconstrained",
              opt::UnconstrainedMarkowitz::global_minimum_variance(mu, sigma, rf).weights);
        print("tangency_unconstrained",
              opt::UnconstrainedMarkowitz::maximum_sharpe_portfolio(mu, sigma, rf).weights);
        for (size_t k = 0; k < targets.size(); ++k) {
            print("target_return_" + std::to_string(k),
                  opt::UnconstrainedMarkowitz::target_return_portfolio(mu, sigma, targets[k], rf).weights);
        }
        print("risk_parity", opt::UnconstrainedMarkowitz::risk_parity_portfolio(mu, sigma, rf).weights);

        opt::ConstrainedQpConfig cfg;
        cfg.min_weight = min_w;
        cfg.max_weight = max_w;
        cfg.risk_free_rate = rf;
        const opt::ConstrainedQpOptimizer qp(cfg);
        print("gmv_constrained", qp.global_minimum_variance(mu, sigma).weights);
        for (size_t k = 0; k < gammas.size(); ++k) {
            print("risk_aversion_" + std::to_string(k),
                  qp.optimize_risk_aversion(mu, sigma, gammas[k]).weights);
        }
        print("max_sharpe_constrained", qp.maximum_sharpe_portfolio(mu, sigma).weights);

        const size_t projections = read_count(in);
        for (size_t k = 0; k < projections; ++k) {
            const Eigen::VectorXd v = read_vector(in, n);
            print("projection_" + std::to_string(k),
                  opt::ConstrainedQpOptimizer::project_onto_bounded_simplex(v, min_w, max_w));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
