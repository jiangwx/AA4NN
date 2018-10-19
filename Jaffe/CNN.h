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

enum layer_type {
    io_,conv_,maxpool_,avgpool_,lrn_,sigmoid_,innerproduct_,batchnorm_,relu_,concat_,defualt_
};

struct layer
{
    layer_type type;
    char name[15];
    int iw, ih, ic, ow, oh, oc;
    int k, s, p;
};

/**********common.cpp************/
void load_fm(float* fm, layer l);
void load_mean(float *mean, layer l);
void load_weight(float *weight, layer l);
void load_bias(float *bias, layer l);
void generate_fm(float* fm, layer l);
void generate_weight(float* weight, layer l);
void show_fm(float* fm, layer l);
void show_matrix(float* matrix, int M, int N);
void trans_matrix(float* in, float* out, int M, int N);
void check_fm(float* fm, layer l);

void maxpool(float *ifm, float *ofm, layer l);
void avgpool(float *ifm, float *ofm, layer l);
void lrn(float *ifm, float *ofm, int local, float alpha, float beta, layer l);

void relu(float *ifm, float *ofm, layer l);
void batchnorm(float* ifm, float* ofm, float* mean, float* variale, layer l);










/**********Inception_v1.cpp************/
void googlenet();








void convolution_mm(float *ifm, float *ofm, float *weight, float *bias, layer l);
void convolution(float *ifm, float *ofm, float *weight, float *bias, layer l);

int test();
#endif //GEMM_CNN_H
