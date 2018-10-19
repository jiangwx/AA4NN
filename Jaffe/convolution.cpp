#include "CNN.h"

void convolution(float *ifm, float *ofm, float *weight, float *bias, layer l)
{
    for (int oh = 0; oh < l.oh; oh++)
    {
        for (int ow = 0; ow < l.ow; ow++)
        {
            for (int oc = 0; oc < l.oc; oc++)
            {
                float odata = 0;
                int outcnt = oc * l.oh*l.ow + oh * l.ow + ow;
                for (int id = 0; id < l.ic; id++)
                {
                    for (int wh = 0; wh < l.k; wh++)
                    {
                        for (int ww = 0; ww < l.k; ww++)
                        {
                            float ret = 0;

                            int xind = ow*l.s - l.p + ww;
                            int yind = oh*l.s - l.p + wh;

                            int incnt = id * l.ih*l.iw + yind *l.iw + xind;
                            int weicnt = oc * l.ic*l.k*l.k + id * l.k* l.k + wh * l.k + ww;

                            if ((xind < 0) || (xind >(l.iw - 1)) || (yind < 0) || (yind >(l.ih - 1)))
                                ret = 0;
                            else
                                ret = ifm[incnt];
                            ret *= weight[weicnt];
                            odata += ret;
                        }
                    }
                }
                odata += bias[oc];
                ofm[outcnt] = odata;
            }
        }
    }
}

