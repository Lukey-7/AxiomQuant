#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace quant::tests {

struct TestFailure {
    std::string test_name;
    std::string file;
    int line;
    std::string message;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void register_test(std::string name, std::function<void()> func) {
        tests_.push_back({std::move(name), std::move(func)});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;
        std::vector<TestFailure> failures;

        std::cout << "\n=========================================================================\n";
        std::cout << "                        RUNNING QUANT ENGINE TESTS                       \n";
        std::cout << "=========================================================================\n";

        for (const auto& [name, func] : tests_) {
            current_test_name_ = name;
            bool test_passed = true;
            try {
                func();
            } catch (const std::exception& e) {
                test_passed = false;
                failures.push_back({name, "Exception", 0, e.what()});
            } catch (...) {
                test_passed = false;
                failures.push_back({name, "Unknown Exception", 0, "Non-std exception thrown"});
            }

            if (test_passed && !test_has_failed_) {
                passed++;
                std::cout << " [PASS] " << name << "\n";
            } else {
                failed++;
                std::cout << " [FAIL] " << name << "\n";
            }
            test_has_failed_ = false;
        }

        std::cout << "-------------------------------------------------------------------------\n";
        std::cout << "Summary: " << passed << " Passed, " << failed << " Failed, Total: " << tests_.size()
                  << "\n";

        if (!failures.empty()) {
            std::cout << "\nFailures:\n";
            for (const auto& f : failures) {
                std::cout << "  - " << f.test_name << " (" << f.file << ":" << f.line << "): " << f.message
                          << "\n";
            }
        }
        std::cout << "=========================================================================\n\n";

        return failed == 0 ? 0 : 1;
    }

    void record_failure(const std::string& file, int line, const std::string& message) {
        test_has_failed_ = true;
        std::cerr << "    FAILED at " << file << ":" << line << ": " << message << "\n";
    }

private:
    struct TestCase {
        std::string name;
        std::function<void()> func;
    };
    std::vector<TestCase> tests_;
    std::string current_test_name_;
    bool test_has_failed_{false};
};

struct TestRegistrar {
    TestRegistrar(const std::string& name, std::function<void()> func) {
        TestRegistry::instance().register_test(name, std::move(func));
    }
};

}   // namespace quant::tests

#define TEST_CASE(name)                                                             \
    static void test_func_##name();                                                 \
    static ::quant::tests::TestRegistrar registrar_##name(#name, test_func_##name); \
    static void test_func_##name()

#define EXPECT_TRUE(cond)                                                                        \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            ::quant::tests::TestRegistry::instance().record_failure(__FILE__, __LINE__,          \
                                                                    "Condition failed: " #cond); \
        }                                                                                        \
    } while (false)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                                            \
    do {                                                                                           \
        if ((a) != (b)) {                                                                          \
            std::ostringstream ss;                                                                 \
            ss << "Expected " #a " == " #b " (" << (a) << " vs " << (b) << ")";                    \
            ::quant::tests::TestRegistry::instance().record_failure(__FILE__, __LINE__, ss.str()); \
        }                                                                                          \
    } while (false)

#define EXPECT_NEAR(a, b, tol)                                                                        \
    do {                                                                                              \
        double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));                      \
        if (diff > (tol)) {                                                                           \
            std::ostringstream ss;                                                                    \
            ss << "Expected " #a " near " #b " (diff=" << diff << " > tol=" << (tol) << ", a=" << (a) \
               << ", b=" << (b) << ")";                                                               \
            ::quant::tests::TestRegistry::instance().record_failure(__FILE__, __LINE__, ss.str());    \
        }                                                                                             \
    } while (false)

#define EXPECT_THROW(stmt, ExceptionType)                                                \
    do {                                                                                 \
        bool caught = false;                                                             \
        try {                                                                            \
            stmt;                                                                        \
        } catch (const ExceptionType&) {                                                 \
            caught = true;                                                               \
        } catch (...) {                                                                  \
        }                                                                                \
        if (!caught) {                                                                   \
            ::quant::tests::TestRegistry::instance().record_failure(                     \
                __FILE__, __LINE__, "Expected exception " #ExceptionType " for " #stmt); \
        }                                                                                \
    } while (false)
