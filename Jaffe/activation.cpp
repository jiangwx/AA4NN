#include "CNN.h"
void relu(float *ifm, float *ofm, layer l)
{
    float odata;
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int i = 0; i < l.ow*l.oh; i++)
        {
            odata = ifm[oc*l.ow*l.oh + i];
            if (odata < 0)
                ofm[oc*l.ow*l.oh + i] = 0;
            else
                ofm[oc*l.ow*l.oh + i] = odata;
        }
    }
}

void batchnorm(float* ifm, float* ofm, float* mean, float* variale, float* bias, layer l)
{
    float odata;
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int i = 0; i < l.ow*l.oh; i++)
        {
            odata = ( ifm[oc*l.ow*l.oh + i] - mean[oc]) * variale[oc] + bias[oc];
            if (odata < 0)
                ofm[oc*l.ow*l.oh + i] = odata * 0.1;
            else
                ofm[oc*l.ow*l.oh + i] = odata;
        }
    }
}
