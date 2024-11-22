#include "utils.h"
#include <assert.h>
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

gsl_matrix **weights_alloc(size_t *shape, size_t n) {

    gsl_matrix **ws = malloc(n * sizeof(gsl_matrix *));

    for (int i = 0; i < n; i++) {
        ws[i] = gsl_matrix_calloc(shape[i + 1], shape[i]);
    }

    return ws;
}

void weights_free(gsl_matrix **ms, size_t n) {

    for (int i = 0; i < n; i++) {
        gsl_matrix_free(ms[i]);
    }

    free(ms);
}

gsl_vector *vector_alloc_from(double *x, size_t n) {
    gsl_vector *v = gsl_vector_alloc(n);

    for (int i = 0; i < n; i++) {
        gsl_vector_set(v, i, x[i]);
    }

    return v;
}

gsl_vector *vector_alloc_from_static(size_t n, double x[]) {
    gsl_vector *v = gsl_vector_alloc(n);

    for (int i = 0; i < n; i++) {
        gsl_vector_set(v, i, x[i]);
    }

    return v;
}

gsl_matrix *weights_alloc_from(double **x, size_t m, size_t n) {
    gsl_matrix *w = gsl_matrix_alloc(m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gsl_matrix_set(w, i, j, x[i][j]);
        }
    }

    return w;
}

gsl_matrix *weights_alloc_from_static(size_t m, size_t n, double x[][n]) {
    gsl_matrix *w = gsl_matrix_alloc(m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gsl_matrix_set(w, i, j, x[i][j]);
        }
    }

    return w;
}

void mvmul(gsl_matrix *w, gsl_vector *a, gsl_vector *b, gsl_vector *z) {
    gsl_vector_memcpy(z, b);
    gsl_blas_dgemv(CblasNoTrans, 1, w, a, 1, z);
}

// This function must be optimized with SIMD.
void vfapply(gsl_vector *dst, gsl_vector *src, double (*f)(double x)) {
    for (size_t i = 0; i < src->size; i++) {
        double y = f(gsl_vector_get(src, i));
        gsl_vector_set(dst, i, y);
    }
}

void vvmul(gsl_vector *a, gsl_vector *b, gsl_matrix *dst) {

    gsl_matrix_view av = gsl_matrix_view_vector(a, a->size, 1);
    gsl_matrix_view bv = gsl_matrix_view_vector(b, b->size, 1);

    gsl_blas_dgemm(
        CblasNoTrans, CblasTrans, 1.0, &av.matrix, &bv.matrix, 1.0, dst);
}

void vhadamard(gsl_vector *dst, const gsl_vector *src) {

    for (int i = 0; i < dst->size; i++) {
        double a = gsl_vector_get(dst, i);
        double b = gsl_vector_get(src, i);
        gsl_vector_set(dst, i, a * b);
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + gsl_sf_exp(-x));
}

double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}
