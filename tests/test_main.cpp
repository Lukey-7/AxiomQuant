#include "test_harness.hpp"

int main() {
    return quant::tests::TestRegistry::instance().run_all();
}
