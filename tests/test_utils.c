#include "../src/utils.h"

#include <criterion/criterion.h>
#include <criterion/new/assert.h>

Test(utils, sigmoid) {
    cr_assert(ieee_ulp_eq(dbl, sigmoid(0), 0.5, 4),
              "Sigmoid of 0 should be 0.5.");
}

Test(utils, vectors_alloc_one) {
    size_t *shape = malloc(1 * sizeof(size_t *));
    shape[0] = 2;

    gsl_vector **vs = vectors_alloc(shape, 1);

    for (int i = 0; i < 1; i++) {
        cr_expect(eq(shape[i], vs[i]->size),
                  "Sizes didn't match when i = %d: %lu != %lu.",
                  i,
                  shape[i],
                  vs[i]->size);
    }

    vectors_free(vs, 1);
    free(shape);
}

Test(utils, vectors_alloc_three) {
    size_t *shape = malloc(3 * sizeof(size_t *));
    shape[0] = 2;
    shape[1] = 4;
    shape[2] = 1;

    gsl_vector **vs = vectors_alloc(shape, 3);

    for (int i = 0; i < 3; i++) {
        cr_expect(eq(shape[i], vs[i]->size),
                  "Sizes didn't match when i = %d: %lu != %lu.",
                  i,
                  shape[i],
                  vs[i]->size);
    }

    vectors_free(vs, 3);
    free(shape);
}

Test(utils, weights_alloc_one) {
    size_t *shape = malloc(2 * sizeof(size_t *));
    shape[0] = 2;
    shape[1] = 4;

    gsl_matrix **ws = weights_alloc(shape, 1);

    cr_expect(eq(ws[0]->size1, shape[1]),
              "Number of rows didn't match: %lu != %lu.",
              ws[0]->size1,
              shape[1]);

    cr_expect(eq(ws[0]->size2, shape[0]),
              "Number of columns didn't match: %lu != %lu.",
              ws[0]->size2,
              shape[0]);

    weights_free(ws, 1);
    free(shape);
}

Test(utils, weights_alloc_two) {
    size_t *shape = malloc(3 * sizeof(size_t *));
    shape[0] = 2;
    shape[1] = 4;
    shape[2] = 1;

    gsl_matrix **ws = weights_alloc(shape, 2);

    for (int i = 0; i < 2; i++) {
        cr_expect(eq(ws[i]->size1, shape[i + 1]),
                  "Number of rows didn't match when i = %d: %lu != %lu.",
                  i,
                  ws[i]->size1,
                  shape[i + 1]);

        cr_expect(eq(ws[i]->size2, shape[i]),
                  "Number of columns didn't match when i = %d: %lu != %lu.",
                  i,
                  ws[i]->size2,
                  shape[i]);
    }

    weights_free(ws, 2);
    free(shape);
}

Test(utils, mvmul_one_by_one) {

    double _w[][1] = {{2}};
    double _a[1] = {3};
    double _b[1] = {1};

    gsl_matrix *w = weights_alloc_from_static(1, 1, _w);
    gsl_vector *a = vector_alloc_from_static(1, _a);
    gsl_vector *b = vector_alloc_from_static(1, _b);
    gsl_vector *z = gsl_vector_calloc(1);

    mvmul(w, a, b, z);

    cr_expect(eq(2, gsl_matrix_get(w, 0, 0)));

    cr_expect(eq(3, gsl_vector_get(a, 0)));

    cr_expect(eq(1, gsl_vector_get(b, 0)));

    cr_expect(eq(7, gsl_vector_get(z, 0)));

    gsl_matrix_free(w);
    gsl_vector_free(a);
    gsl_vector_free(b);
    gsl_vector_free(z);
}

Test(utils, mvmul_two_by_two) {

    double _w[][2] = {{1, 2}, {3, 4}};
    double _a[2] = {5, 6};
    double _b[2] = {7, 8};

    gsl_matrix *w = weights_alloc_from_static(2, 2, _w);
    gsl_vector *a = vector_alloc_from_static(2, _a);
    gsl_vector *b = vector_alloc_from_static(2, _b);
    gsl_vector *z = gsl_vector_calloc(2);

    mvmul(w, a, b, z);

    for (int i = 0; i < 2; i++) {

        for (int j = 0; j < 2; j++) {
            cr_expect(eq(_w[i][j], gsl_matrix_get(w, i, j)),
                      "The matrix w should stay the same.");
        }

        cr_expect(eq(_a[i], gsl_vector_get(a, i)),
                  "The vector a should stay the same.");
        cr_expect(eq(_b[i], gsl_vector_get(b, i)),
                  "The vector b should stay the same.");
    }

    cr_expect(eq(24, gsl_vector_get(z, 0)), "The values should be the same.");
    cr_expect(eq(47, gsl_vector_get(z, 1)), "The values should be them same.");

    gsl_matrix_free(w);
    gsl_vector_free(a);
    gsl_vector_free(b);
    gsl_vector_free(z);
}

double square(double x) {
    return x * x;
}

Test(utils, vfapply) {

    double _x[] = {0, 1, 2, 3, 4};

    gsl_vector *x = vector_alloc_from_static(5, _x);
    gsl_vector *y = gsl_vector_alloc(5);

    vfapply(y, x, square);

    for (int i = 0; i < 5; i++) {
        cr_expect(ieee_ulp_eq(dbl, _x[i], gsl_vector_get(x, i), 4),
                  "The vector x should stay the same.");

        cr_expect(ieee_ulp_eq(dbl, _x[i] * _x[i], gsl_vector_get(y, i), 4),
                  "The values should be the same.");
    }

    gsl_vector_free(x);
    gsl_vector_free(y);
}
