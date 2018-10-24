#include <cmath>
#include "CNN.h"

void lrn(float *ifm, float *ofm, int local, float alpha, float beta, layer l)
{
    int pad = (local - 1) / 2;
    int len = l.ic*l.ih*l.iw;
    float *scale = (float *)malloc(sizeof(float)*len);
    float *padded_square = (float *)malloc(sizeof(float)*(l.ic + local - 1)*l.ih*l.iw);
    float alpha_over_size = alpha / local;
    for (int i = 0; i < len + 2 * pad*l.ih*l.iw; i++) padded_square[i] = 0;
    for (int i = 0; i < len; i++) scale[i] = 1;
    for (int i = 0; i < len; i++) padded_square[i + pad*l.ih*l.iw] = alpha_over_size * ifm[i] * ifm[i];

    for (int d = 0; d <local; d++)
    {//caffe_axpy<Dtype>
        for (int i = 0; i<l.ih*l.iw; ++i)scale[i] += padded_square[i + d*l.ih*l.iw];
    }

    for (int d = 1; d<l.ic; d++)
    {	//caffe_copy<Dtype>
        for (int i = 0; i < l.ih*l.iw; ++i)  scale[i + d*l.ih*l.iw]  = scale[i + (d-1)*l.ih*l.iw];
        for (int i = 0; i < l.ih*l.iw; i++)  scale[i + d*l.ih*l.iw] += padded_square[i + (d + local - 1)*l.ih*l.iw];
        for (int i = 0; i < l.ih*l.iw; i++)  scale[i + d*l.ih*l.iw] -= padded_square[i + (d - 1)*l.ih*l.iw];
    }
    for (int i = 0; i < len; i++) ofm[i] = powf(scale[i],-beta) ;
    for (int i = 0; i < len; i++) ofm[i] = ofm[i]*ifm[i];
    free(scale);
    free(padded_square);
}
