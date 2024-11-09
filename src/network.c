#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>

typedef struct network {
    size_t n;
    size_t *shape;
    gsl_matrix **weights;
    gsl_vector **biases;
} network_t;

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

network_t *network_alloc(size_t *shape, size_t n) {

    network_t *nw = malloc(sizeof(network_t));

    nw->n = n;
    nw->shape = shape;
    nw->weights = matrices_alloc(shape, n);
    nw->biases = vectors_alloc(shape, n);

    return nw;
}

void network_free(network_t *nw) {

    for (int i = 0; i < nw->n - 1; i++) {
        gsl_matrix_free(nw->weights[i]);
        gsl_vector_free(nw->biases[i]);
    }

    free(nw->shape);
    free(nw);
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

void forward(network_t *nw, gsl_vector **as, gsl_vector **zs, gsl_vector *x) {

    // Set the input to the first layer.
    as[0] = x;

    for (int i = 0; i < nw->n; i++) {
        mvmul(nw->weights[i], as[i], nw->biases[i], zs[i]);
        vfapply(as[i + 1], zs[i], sigmoid);
    }
}

void backpropagation(network_t *nw, gsl_vector *x, gsl_vector *y) {

    gsl_vector **as = vectors_alloc(nw->shape, nw->n);
    gsl_vector **zs = vectors_alloc(nw->shape + 1, nw->n - 1);

    forward(nw, as, zs, x);

    vectors_free(as, nw->n);
    vectors_free(zs, nw->n - 1);
}
