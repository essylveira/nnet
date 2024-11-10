#include "utils.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_exp.h>

gsl_vector **vectors_alloc(size_t *shape, size_t n) {

    gsl_vector **vs = malloc(n * sizeof(gsl_vector *));

    for (int i = 0; i < n; i++) {
        vs[i] = gsl_vector_calloc(shape[i]);
    }

    return vs;
}

void vectors_free(gsl_vector **vs, size_t n) {

    for (int i = 0; i < n; i++) {
        gsl_vector_free(vs[i]);
    }

    free(vs);
}

gsl_matrix **matrices_alloc(size_t *shape, size_t n) {

    gsl_matrix **ms = malloc(n * sizeof(gsl_matrix *));

    for (int i = 0; i < n - 1; i++) {
        ms[i] = gsl_matrix_calloc(shape[i + 1], shape[i]);
    }

    return ms;
}

void matrices_free(gsl_matrix **ms, size_t n) {

    for (int i = 0; i < n; i++) {
        gsl_matrix_free(ms[i]);
    }

    free(ms);
}

void mvmul(gsl_matrix *w, gsl_vector *a, gsl_vector *b, gsl_vector *z) {
    gsl_vector_memcpy(z, b);
    gsl_blas_dgemv(CblasNoTrans, 1, w, a, 1, z);
}

// This function must be optimized with SIMD.
void vfapply(gsl_vector *dst, gsl_vector *src, double (*f)(double x)) {
    for (size_t i = 0; i < src->size; i++) {
        double y = f(gsl_vector_get(dst, i));
        gsl_vector_set(src, i, y);
    }
}

double sigmoid(double x) { return 1.0 / (1.0 + gsl_sf_exp(-x)); }
