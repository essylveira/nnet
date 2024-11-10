#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>

#include "network.h"
#include "utils.h"

struct network {
    size_t n;
    size_t *shape;
    gsl_matrix **weights;
    gsl_vector **biases;
};

network_t *network_alloc(size_t *shape, size_t n) {

    network_t *nw = malloc(sizeof(network_t));

    nw->n = n;
    nw->shape = shape;
    nw->weights = weights_alloc(shape, n);
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

void network_set_input(network_t *nw, double **x, size_t m, size_t n) {
}

void network_set_output(network_t *nw, double **y, size_t m, size_t n) {
}

void network_train(network_t *nw) {
}

void network_predict(network_t *nw, double *x, size_t n) {
}
