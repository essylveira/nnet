#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_matrix.h>
#include <stdlib.h>

gsl_vector **vectors_alloc(size_t *shape, size_t n);
void vectors_free(gsl_vector **vs, size_t n);
gsl_matrix **weights_alloc(size_t *shape, size_t n);
void weights_free(gsl_matrix **ms, size_t n);

gsl_vector *vector_alloc_from(double *x, size_t n);
gsl_vector *vector_alloc_from_static(size_t n, double x[]);
gsl_matrix *weights_alloc_from(double **x, size_t m, size_t n);
gsl_matrix *weights_alloc_from_static(size_t m, size_t n, double [][n]);

void mvmul(gsl_matrix *w, gsl_vector *a, gsl_vector *b, gsl_vector *z);
void vfapply(gsl_vector *dst, gsl_vector *src, double (*f)(double x));

double sigmoid(double x);

#endif // UTILS_H
