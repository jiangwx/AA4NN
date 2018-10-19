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
    AAF *ifm_AA = new AAF[l.ic*l.ih*l.iw];
    float* ofm = (float*)calloc(l.oc*l.oh*l.ow, sizeof(float));
    AAF *ofm_AA= new AAF[l.oc*l.oh*l.ow];
    float* weight = (float*)calloc(l.ic*l.oc*l.k*l.k, sizeof(float));
    float* bias = (float*)calloc(l.oc, sizeof(float));

    generate_fm(ifm,l);
    add_AA(ifm,ifm_AA,l);
    generate_weight(weight,l);
    generate_bias(bias,l);


    convolution(ifm,ofm,weight,bias,l);
    AA_convolution(ifm_AA,ofm_AA,weight,bias,l);
    check_fm(ofm, ofm_AA, l);

    free(ifm);
    delete []ifm_AA;
    free(ofm);
    delete []ofm_AA;
    free(weight);
    free(bias);

    return 0;
}