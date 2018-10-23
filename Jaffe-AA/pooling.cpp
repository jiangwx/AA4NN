#include "CNN.h"

void maxpool(float *ifm, float *ofm, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                float odata = -34000000;
                for(int kh = 0; kh < l.k; kh++)
                {
                    for(int kw = 0; kw < l.k; kw++)
                    {
                        float ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw >(l.iw - 1)) || (fh < 0) || (fh >(l.ih - 1)))
                            ret = 34000000;
                        else
                            ret = ifm[fm_index];

                        if (ret > odata) odata = ret;
                    }
                }
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata;
            }
        }
    }
}

void AA_maxpool(AAF *ifm, AAF *ofm, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                AAF odata = -34000000;
                for(int kh = 0; kh < l.k; kh++)
                {
                    for(int kw = 0; kw < l.k; kw++)
                    {
                        AAF ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw >(l.iw - 1)) || (fh < 0) || (fh >(l.ih - 1)))
                            ret = 0;
                        else
                            ret = ifm[fm_index];

                        if (ret.get_center() > odata.get_center()) odata = ret;
                    }
                }
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata;
            }
        }
    }
}

void avgpool(float *ifm, float *ofm, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                float odata = 0;
                for(int kh=0;kh<l.k;kh++)
                {
                    for(int kw=0;kw<l.k;kw++)
                    {
                        float ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw > (l.iw - 1)) || (fh < 0) || (fh > (l.ih - 1)))
                            ret = 0;
                        else
                            ret = ifm[fm_index];

                        odata = odata + ret;
                    }
                }
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata / (l.k*l.k);
            }
        }
    }
}

void AA_avgpool(AAF *ifm, AAF *ofm, layer l)
{
    int kernel_size=l.k * l.k;
    float _kernel_size=1/(float)kernel_size;
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                AAF odata = 0;
                AAF odata_temp = 0;
                for(int kh=0;kh<l.k;kh++)
                {
                    for(int kw=0;kw<l.k;kw++)
                    {
                        AAF ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw > (l.iw - 1)) || (fh < 0) || (fh > (l.ih - 1)))
                            ret = 0;
                        else
                            ret = ifm[fm_index];

                        odata = odata + ret;
                    }
                }
                odata_temp = odata*_kernel_size;
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata_temp;
            }
        }
    }
}