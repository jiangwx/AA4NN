#include "CNN.h"
#define CheckScale 0.01

void load_fm(float* fm, layer l)
{
    int len = l.ow* l.oh*l.oc;

    char nstr[50];

    sprintf(nstr, "../blobs/%s.bb", l.name);

    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, len * sizeof(float), fp);
    fclose(fp);
}

void load_mean(float *mean, layer l)
{
    char nstr[50];
    FILE *fp;

    fp = fopen(nstr, "rb");
    sprintf(nstr, "/mnt/weight/%s.mn", l.name);
    fp = fopen(nstr, "rb");
    fread(mean, 1, l.oc * sizeof(float), fp);
    fclose(fp);

    sprintf(nstr, "/mnt/weight/%s.vs", l.name);
    fp = fopen(nstr, "rb");
    fread(&mean[l.oc], 1, l.oc * sizeof(float), fp);
    fclose(fp);

    sprintf(nstr, "/mnt/weight/%s.bs", l.name);
    fp = fopen(nstr, "rb");
    fread(&mean[2 * l.oc], 1, l.oc * sizeof(float), fp);
    fclose(fp);

}

void load_weight(float *weight, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.wt", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, l.ic*l.oc*l.k*l.k * sizeof(float), fp);
    fclose(fp);
}

void load_bias(float *bias, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.bs", l.name);
    FILE *fp = fopen(nstr, "rb");
    fp = fopen(nstr, "rb");
    fread(bias, 1, l.oc * sizeof(float), fp);
    fclose(fp);
}

void generate_fm(float* fm, layer l)
{
    for (int i = 0; i < l.ic*l.iw*l.ih; i++) fm[i] = i;
}

void generate_weight(float* weight, layer l)
{
    for (int i = 0; i < l.ic*l.oc*l.k*l.k; i++) weight[i] = i;
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

void check_fm(float* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    float *tmp = (float *)malloc(sizeof(float)*len);

    char nstr[50];
    sprintf(nstr, "../blobs/%s.bb", l.name);

    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len * sizeof(float), fp);
    fclose(fp);

    static int err = 0;

    for (int j = 0; j < len; j++)
    {
        if (((fm[j] - tmp[j]) > CheckScale) || ((fm[j] - tmp[j]) < -CheckScale))
        {
            err++;
            printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("error cnt= %d, %s\n", err, l.name);
    else
        printf("correct %s \n", l.name);

    free(tmp);
}

