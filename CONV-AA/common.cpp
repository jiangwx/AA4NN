#include "CNN.h"

void generate_fm(float* fm, layer l)
{
    for (int i = 0; i < l.ic*l.iw*l.ih; i++) fm[i] = i;
}

void generate_weight(float* weight, layer l)
{
    for (int i = 0; i < l.ic*l.oc*l.k*l.k; i++) weight[i] = i;
}

void generate_bias(float* bias, layer l)
{
    for (int i = 0; i < l.oc; i++) bias[i] = i;
}

void show_fm(float* fm, layer l)
{
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                std::cout << fm[i]<<", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void show_matrix(float* matrix, int M, int N)
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            std::cout<<matrix[i*N+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void trans_matrix(float* in, float* out, int M, int N)//M: the origin rows, N: the origin cols
{
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            out[j*M+i]=in[i*N+j];
        }

    }
}

void check_fm(float* golden, AAF* test, layer l)
{
    int err = 0;
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                if (abs((golden[i] - (float)test[i].get_center())) > 1)
                {
                    err++;
                    printf("index: %d, golden: %f, test %f\n",i,golden[i],(float)test[i].get_center());
                    //std::cout << "index:" << i <<" golden: " << golden[i] <<" test: " << (float)test[i].get_center() << std::endl;
                }
            }
        }
    }
    if (err)std::cout << " FAILED!!!" << std::endl;
    else  std::cout << "  PASSED!!!" << std::endl;
}