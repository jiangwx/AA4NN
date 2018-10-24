#ifndef CNN_H
#define CNN_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <cblas.h>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define img_w 224
#define img_h 224

enum layer_type 
{
    io_,conv_,maxpool_,avgpool_,lrn_,sigmoid_,innerproduct_,batchnorm_,relu_,concat_,softmax_,defualt_
};

struct layer
{
    layer_type type;
    char name[15];
    int iw, ih, ic, ow, oh, oc;
    int k, s, p;
};

/**********common.cpp************/
void load_image(char* img_path, float* blob,float* mean);
void load_mean_image(char* img_path, float* blob);
void load_fm(float* fm, layer l, char* Net);
void load_mean(float *mean, layer l, char* Net);
void load_weight(float *weight, layer l, char* Net);
void load_bias(float *bias, layer l, char* Net);
void generate_fm(float* fm, layer l);
void generate_weight(float* weight, layer l);
void show_fm(float* fm, layer l);
void show_matrix(float* matrix, int M, int N);
void trans_matrix(float* in, float* out, int M, int N);
void check_fm(float* fm, layer l, char* Net);



void convolution(float *ifm, float *ofm, float *weight, float *bias, layer l);
void convolution_mm(float *ifm, float *ofm, float *weight, float* bias, layer l);

void innerproduct(float *ifm, float *ofm, float *weight, float *bias, layer l);

void maxpool(float *ifm, float *ofm, layer l);

void avgpool(float *ifm, float *ofm, layer l);

void lrn(float *ifm, float *ofm, int local, float alpha, float beta, layer l);

void relu(float *ifm, float *ofm, layer l);

void batchnorm(float* ifm, float* ofm, float* mean, float* variale, float* bias, layer l);

#endif //CNN_H
