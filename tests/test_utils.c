#include "../src/utils.h"
#include <criterion/criterion.h>

Test(utils, sigmoid) {
    cr_assert_eq(sigmoid(0), 0.5, "Sigmoid of 0 should be 0.5.");
}

Test(utils, vectors_alloc) {
}

Test(utils, matrices_alloc) {
}
