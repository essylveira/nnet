#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct matrix {
    double **values;
    int m, n;
} matrix_t;

typedef struct network {
    int layern;
    int *sizes;
    matrix_t **biases;
    matrix_t **weights;
} network_t;

matrix_t *matrix_create(int m, int n) {

    matrix_t *mat = malloc(sizeof(matrix_t));

    mat->values = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        mat->values[i] = calloc(n, sizeof(double));
    }

    mat->m = m;
    mat->n = n;

    return mat;
}

matrix_t *matrix_ones(int m, int n) {

    matrix_t *ones = matrix_create(m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ones->values[i][j] = 1;
        }
    }

    return ones;
}

void matrix_set_values(matrix_t *mat, double **values) { mat->values = values; }

void matrix_show(matrix_t *mat) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            printf(" %.6f", mat->values[i][j]);
        }
        printf("\n");
    }
}

matrix_t *randn(int l, int c) {

    matrix_t *mat = matrix_create(l, c);

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < c; j++) {
            mat->values[i][j] = (double)rand() / RAND_MAX;
        }
    }

    return mat;
}

network_t *network_create(int *sizes, int n) {

    network_t *nw = malloc(sizeof(network_t));

    nw->layern = n;
    nw->sizes = sizes;
    nw->biases = malloc((n - 1) * sizeof(matrix_t *));
    nw->weights = malloc((n - 1) * sizeof(matrix_t *));

    for (int i = 0; i < nw->layern - 1; i++) {
        nw->biases[i] = randn(sizes[i + 1], 1);
    }

    int i = 0;
    while (i < nw->layern - 1) {
        nw->weights[i] = randn(sizes[i + 1], sizes[i]);
        i++;
    }

    return nw;
}

void network_show(network_t *nw) {
    
    for (int i = 0; i < nw->layern - 1; i++) {
        matrix_show(nw->weights[i]);
        matrix_show(nw->biases[i]);
    }
}

double sigmoid(double z) { return (double)1.0 / (1.0 + exp(-z)); }

matrix_t *matrix_sigmoid(matrix_t *mat) {

    matrix_t *t = matrix_create(mat->m, mat->n);

    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            t->values[i][j] = sigmoid(mat->values[i][j]);
        }
    }

    return t;
}

double dot(double *l, double *r, int n) {

    double acc = 0;

    for (int i = 0; i < n; i++) {
        acc += l[i] * r[i];
    }

    return acc;
}

double *column(matrix_t *mat, int j) {
    double *column = malloc(mat->m * sizeof(double));

    for (int i = 0; i < mat->m; i++) {
        column[i] = mat->values[i][j];
    }

    return column;
}

matrix_t *matrix_mul(matrix_t *a, matrix_t *b) {

    matrix_t *c = matrix_create(a->m, b->n);

    assert(a->n == b->m);

    for (int i = 0; i < a->m; i++) {
        for (int j = 0; j < b->n; j++) {
            c->values[i][j] = dot(a->values[i], column(b, j), a->n);
        }
    }

    return c;
}

matrix_t *matrix_add(matrix_t *a, matrix_t *b) {
    matrix_t *c = matrix_create(a->m, a->n);

    for (int i = 0; i < a->m; i++) {
        for (int j = 0; j < a->n; j++) {
            c->values[i][j] = a->values[i][j] + b->values[i][j];
        }
    }

    return c;
}

matrix_t *matrix_sub(matrix_t *a, matrix_t *b) {
    matrix_t *c = matrix_create(a->m, a->n);

    for (int i = 0; i < a->m; i++) {
        for (int j = 0; j < a->n; j++) {
            c->values[i][j] = a->values[i][j] - b->values[i][j];
        }
    }

    return c;
}

matrix_t *matrix_hadamard(matrix_t *a, matrix_t *b) {

    matrix_t *h = matrix_create(a->m, a->n);

    for (int i = 0; i < a->m; i++) {
        for (int j = 0; j < a->n; j++) {
            h->values[i][j] = a->values[i][j] * b->values[i][j];
        }
    }

    return h;
}

matrix_t *matrix_sigmoid_prime(matrix_t *mat) {
    matrix_t *ones = matrix_ones(mat->m, mat->n);
    matrix_t *sigmoid = matrix_sigmoid(mat);
    return matrix_hadamard(sigmoid, matrix_sub(ones, sigmoid));
}

matrix_t *feedforward(network_t *nw, matrix_t *input) {

    for (int i = 0; i < nw->layern - 1; i++) {
        input = matrix_mul(nw->weights[i], input);
        input = matrix_add(input, nw->biases[i]);
        input = matrix_sigmoid(input);
    }

    return input;
}

matrix_t **make_input(int n) {
    matrix_t **input = malloc(n * sizeof(matrix_t *));

    input[0] = matrix_create(2, 1);
    input[0]->values[0][0] = 0;
    input[0]->values[1][0] = 0;

    input[1] = matrix_create(2, 1);
    input[1]->values[0][0] = 0;
    input[1]->values[1][0] = 1;

    input[2] = matrix_create(2, 1);
    input[2]->values[0][0] = 1;
    input[2]->values[1][0] = 0;

    input[3] = matrix_create(2, 1);
    input[3]->values[0][0] = 1;
    input[3]->values[1][0] = 1;

    return input;
}

matrix_t **make_output(matrix_t **input, int n) {
    matrix_t **output = malloc(n * sizeof(matrix_t *));

    for (int i = 0; i < n; i++) {
        output[i] = matrix_create(1, 1);
        output[i]->values[0][0] =
            ((int)input[i]->values[0][0]) ^ ((int)input[i]->values[1][0]);
    }

    return output;
}

matrix_t *matrix_copy_shape(matrix_t *mat) {
    matrix_t *copy = matrix_create(mat->m, mat->n);
    return copy;
}

matrix_t *matrix_transposed(matrix_t *mat) {

    matrix_t *transposed = matrix_create(mat->n, mat->m);

    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            transposed->values[j][i] = mat->values[i][j];
        }
    }

    return transposed;
}

matrix_t *matrix_scalar_mul(matrix_t *mat, double scalar) {

    matrix_t *mat_scaled = matrix_create(mat->m, mat->n);

    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            mat_scaled->values[i][j] = scalar * mat->values[i][j];
        }
    }

    return mat_scaled;
}

matrix_t *cost_derivative(matrix_t *y, matrix_t *yhat) {
    return matrix_sub(yhat, y);
}

void backpropagation(network_t *nw, matrix_t *x, matrix_t *y,
                     matrix_t ***_nabla_w, matrix_t ***_nabla_b) {

    int n = nw->layern - 1;

    matrix_t **as = malloc((n + 1) * sizeof(matrix_t *));
    matrix_t **zs = malloc(n * sizeof(matrix_t *));

    as[0] = x;

    for (int i = 0; i < n; i++) {
        zs[i] = matrix_add(matrix_mul(nw->weights[i], as[i]), nw->biases[i]);
        as[i + 1] = matrix_sigmoid(zs[i]);
    }

    matrix_t *d = matrix_hadamard(cost_derivative(y, as[n]),
                                  matrix_sigmoid_prime(zs[n - 1]));

    matrix_t **nabla_b = malloc(n * sizeof(matrix_t *));
    matrix_t **nabla_w = malloc(n * sizeof(matrix_t *));

    nabla_b[n - 1] = d;
    nabla_w[n - 1] = matrix_mul(d, matrix_transposed(as[n - 1]));

    for (int i = 1; i < n; i++) {
        d = matrix_hadamard(
            matrix_mul(matrix_transposed(nw->weights[n - i]), d),
            matrix_sigmoid_prime(zs[n - 1 - i]));
        nabla_b[n - 1 - i] = d;
        nabla_w[n - 1 - i] = matrix_mul(d, matrix_transposed(as[n - 1 - i]));
    }

    *_nabla_w = nabla_w;
    *_nabla_b = nabla_b;
}

void update(network_t *nw, matrix_t **x, matrix_t **y, double eta) {
    matrix_t **nabla_w;
    matrix_t **nabla_b;

    for (int i = 0; i < 4; i++) {
        backpropagation(nw, x[i], y[i], &nabla_w, &nabla_b);

        for (int j = 0; j < nw->layern - 1; j++) {
            nw->weights[j] =
                matrix_add(nw->weights[j], matrix_scalar_mul(nabla_w[j], -eta / 4));
            nw->biases[j] =
                matrix_add(nw->biases[j], matrix_scalar_mul(nabla_b[j], -eta / 4));
        }
    }
}

int main() {

    srand(0);

    matrix_t **x = make_input(0);
    matrix_t **y = make_output(x, 4);

    int sizes[] = {2, 2, 1};

    network_t *nw = network_create(sizes, 3);

    network_show(nw);

    update(nw, x, y, 0.01);

    network_show(nw);

    for (int i = 0; i < 2000; i++) {
        update(nw, x, y, 1);
    }

    network_show(nw);

    for (int i = 0; i < 4; i++) {
        printf("-----------------\n");
        matrix_t *o = feedforward(nw, x[i]);
        matrix_show(x[i]);
        matrix_show(o);
    }

}
