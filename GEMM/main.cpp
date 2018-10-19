#include "CNN.h"

int main()
{
    timeval start,end;

    layer l;
    l.ic = 4;
    l.ih = 56;
    l.iw = 56;
    l.oc = 4;
    l.oh = 56;
    l.ow = 56;
    l.k = 3;
    l.s = 2;
    l.p = 1;

    float* ifm = (float*)calloc(l.ic*l.ih*l.iw, sizeof(float));
    float* ofm_golden = (float*)calloc(l.oc*l.oh*l.ow, sizeof(float));
    float* ofm = (float*)calloc(l.oc*l.oh*l.ow, sizeof(float));
    float* weight = (float*)calloc(l.ic*l.oc*l.k*l.k, sizeof(float));
    float* bias = (float*)calloc(l.oc, sizeof(float));

    generate_fm(ifm,l);
    generate_weight(weight,l);
    generate_bias(bias,l);

    gettimeofday(&start, NULL);
    convolution(ifm,ofm_golden,weight,bias,l);
    gettimeofday(&end, NULL);

    long us = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    printf("convolution took %lu us\n", us);

    gettimeofday(&start, NULL);
    convolution_mm(ifm,ofm,weight,bias,l);
    gettimeofday(&end, NULL);

    us = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    printf("gemm convolution took %lu us\n", us);

    check_fm(ofm_golden,ofm,l);

    free(ifm);
    free(ofm_golden);
    free(ofm);
    free(weight);

    return 0;
}