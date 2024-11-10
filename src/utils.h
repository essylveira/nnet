#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_matrix.h>
#include <stdlib.h>

gsl_vector **vectors_alloc(size_t *shape, size_t n);
void vectors_free(gsl_vector **vs, size_t n);
gsl_matrix **matrices_alloc(size_t *shape, size_t n);
void matrices_free(gsl_matrix **ms, size_t n);

void mvmul(gsl_matrix *w, gsl_vector *a, gsl_vector *b, gsl_vector *z);
void vfapply(gsl_vector *dst, gsl_vector *src, double (*f)(double x));

double sigmoid(double x);

#endif // UTILS_H
