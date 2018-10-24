#include "CNN.h"

void innerproduct(float *ifm, float *ofm, float *weight, float *bias, layer l)
{
	for (int oc = 0; oc < l.oc; oc++)
	{
		float odata = 0;
		for (int ic = 0; ic < l.ic; ic++)
		{
			odata += weight[oc * l.ic + ic] * ifm[ic];
		}
		ofm[oc] = odata+bias[oc];
	}
}
