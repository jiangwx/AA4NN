#ifndef GEMM_CNN_H
#define GEMM_CNN_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cblas.h>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
struct layer
{
    char name[15];
    int iw, ih, ic, ow, oh, oc;
    int k, s, p;
};

void check_fm(float* golden, float* test, layer l);
void show_fm(float* fm, layer l);
void show_matrix(float* matrix, int M, int N);
void trans_matrix(float* in, float* out, int M, int N);
void generate_fm(float* fm, layer l);
void generate_weight(float* weight, layer l);
void generate_bias(float* bias, layer l);

void convolution_mm(float *ifm, float *ofm, float *weight, float* bias, layer l);
void convolution(float *ifm, float *ofm, float *weight, float* bias, layer l);

int test();
#endif //GEMM_CNN_H