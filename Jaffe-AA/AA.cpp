#include "CNN.h"

void AA_add(float* input, AAF *output, layer l)
{
    interval u(-0.00001, 0.00001);
    AAF *aa_index = new AAF[l.oc];

    for(int i = 0; i < l.oc; i++)
    {
        aa_index[i] = u;
    }

    for(int i = 0; i < l.oc; i++)
    {
        for(int j = 0; j < l.ow*l.oh; j++)
        {
            output[i*l.ow*l.oh+j] = input[i*l.ow*l.oh+j] + aa_index[i];
        }
    }
}

void AA_remove(AAF *input, float* output, layer l)
{
    for(int i = 0; i < l.oc; i++)
    {
        for(int j = 0; j < l.ow*l.oh; j++)
        {
            output[i*l.ow*l.oh+j] = (float)input[i*l.ow*l.oh+j].get_center();
        }
    }
}

