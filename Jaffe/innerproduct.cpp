#include "CNN.h"

void innerproduct(float *ifm, float *weight, float *bias, layer l)
{
    const float alpha = 1;
    const float beta = 1;
    int M = l.oc;
    int N = 1;
    int K = l.ic*l.iw*l.ih;
    int lda=K;//A的列
    int ldb=N;//B的列
    int ldc=N;//C的列

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, weight, lda, ifm, ldb, beta, bias, ldc);

}
