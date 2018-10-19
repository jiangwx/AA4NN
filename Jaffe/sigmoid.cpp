#include <cmath>
#include "CNN.h"

void sigmoid(float *ifm, float *ofm, layer l)
{
    for (int i = 0; i < l.ic*l.ih*l.iw; i++) ofm[i] =1/(1+exp(-ifm[i]));
}