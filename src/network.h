#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

typedef struct network network_t;

network_t *network_alloc(size_t *shape, size_t n);
void network_free(network_t *nw);
void network_set_input(network_t *nw, double **x, size_t m, size_t n);
void network_set_output(network_t *nw, double **y, size_t m, size_t n);
void network_train(network_t *nw);
void network_predict(network_t *nw, double *x, size_t n);

#endif // NETWORK_H
