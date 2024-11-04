#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>

typedef struct network {
    size_t n;
    size_t *shape;
    gsl_matrix **weights;
    gsl_vector **biases;
} network_t;

gsl_vector **vectors_alloc(size_t *shape, size_t n) {

    gsl_vector **vs = malloc(n * sizeof(gsl_vector *));

    for (int i = 0; i < n - 1; i++) {
        vs[i] = gsl_vector_calloc(shape[i + 1]);
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

void matrix_vector_mul(gsl_matrix *a, gsl_vector *x, gsl_vector *y) {
    gsl_blas_dgemv(CblasNoTrans, 1, a, x, 1, y);
}

void backpropagation(network_t *nw, gsl_vector *x, gsl_vector *y) {

    gsl_vector **as = vectors_alloc(nw->shape, nw->n);
    gsl_vector **zs = vectors_alloc(nw->shape + 1, nw->n - 1);

    as[0] = x;

    for (int i = 0; i < nw->n; i++) {
        matrix_vector_mul(nw->weights[i], as[i], zs[i]);
    }

    vectors_free(as, nw->n);
    vectors_free(zs, nw->n - 1);
}

int main() {

    gsl_matrix *a = gsl_matrix_alloc(2, 2);
    gsl_matrix *b = gsl_matrix_alloc(2, 2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            gsl_matrix_set(a, i, j, i + j);
            gsl_matrix_set(b, i, j, i - j);
        }
    }

    gsl_matrix_add(a, b);

    gsl_matrix_free(a);
    gsl_matrix_free(b);
}
