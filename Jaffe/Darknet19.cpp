#include "Darknet19.h"
layer DNet[26] = {
        { conv_,"image",     224,224,3,224,224,3,0,0,0 },
        { conv_,"conv1",     224,224,3,224,224,32,3,1,1 },
        { maxpool_,"pool1",  224,224,32,112,112,32,2,2,0 },
        { conv_,"conv2",     112,112,32,112,112,64,3,1,1 },
        { maxpool_,"pool2",  112,112,64,56,56,64,2,2,0 },
        { conv_,"conv3",     56,56,64,56,56,128,3,1,1 },
        { conv_,"conv4",     56,56,128,56,56,64,1,1,0 },
        { conv_,"conv5",     56,56,64,56,56,128,3,1,1 },
        { maxpool_,"pool5",  56,56,128,28,28,128,2,2,0 },
        { conv_,"conv6",     28,28,128,28,28,256,3,1,1 },
        { conv_,"conv7",     28,28,256,28,28,128,1,1,0 },
        { conv_,"conv8",     28,28,128,28,28,256,3,1,1 },
        { maxpool_,"pool8",  28,28,256,14,14,256,2,2,0 },
        { conv_,"conv9",     14,14,256,14,14,512,3,1,1 },
        { conv_,"conv10",    14,14,512,14,14,256,1,1,0 },
        { conv_,"conv11",    14,14,256,14,14,512,3,1,1 },
        { conv_,"conv12",    14,14,512,14,14,256,1,1,0 },
        { conv_,"conv13",    14,14,256,14,14,512,3,1,1 },
        { maxpool_,"pool13", 14,14,512,7,7,512,2,2,0 },
        { conv_,"conv14",    7,7,512,7,7,1024,3,1,1 },
        { conv_,"conv15",    7,7,1024,7,7,512,1,1,0 },
        { conv_,"conv16",    7,7,512,7,7,1024,3,1,1 },
        { conv_,"conv17",    7,7,1024,7,7,512,1,1,0 },
        { conv_,"conv18",    7,7,512,7,7,30,3,1,1 },
        { avgpool_,"pool18", 7,7,30,1,1,30,7,1,0 },
        { softmax_,"softmax",1,1,30,1,1,30,0,0,0 }
};

enum Darknet19_idx {
    image, conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, conv6,
    conv7, conv8, pool8, conv9, conv10, conv11, conv12, conv13, pool13,
    conv14, conv15, conv16, conv17, conv18, pool18, softmax
};

static char Name[]="Darknet19";

static float* image_blob;
static float* conv1_blob;
static float* conv1_weight;
static float* conv1_bn;
static float* pool1_blob;
static float* conv2_blob;
static float* conv2_weight;
static float* conv2_bn;
static float* pool2_blob;
static float* conv3_blob;
static float* conv3_weight;
static float* conv3_bn;
static float* conv4_blob;
static float* conv4_weight;
static float* conv4_bn;
static float* conv5_blob;
static float* conv5_weight;
static float* conv5_bn;
static float* pool5_blob;
static float* conv6_blob;
static float* conv6_weight;
static float* conv6_bn;
static float* conv7_blob;
static float* conv7_weight;
static float* conv7_bn;
static float* conv8_blob;
static float* conv8_weight;
static float* conv8_bn;
static float* pool8_blob;
static float* conv9_blob;
static float* conv9_weight;
static float* conv9_bn;
static float* conv10_blob;
static float* conv10_weight;
static float* conv10_bn;
static float* conv11_blob;
static float* conv11_weight;
static float* conv11_bn;
static float* conv12_blob;
static float* conv12_weight;
static float* conv12_bn;
static float* conv13_blob;
static float* conv13_weight;
static float* conv13_bn;
static float* pool13_blob;
static float* conv14_blob;
static float* conv14_weight;
static float* conv14_bn;
static float* conv15_blob;
static float* conv15_weight;
static float* conv15_bn;
static float* conv16_blob;
static float* conv16_weight;
static float* conv16_bn;
static float* conv17_blob;
static float* conv17_weight;
static float* conv17_bn;
static float* conv18_blob;
static float* conv18_weight;
static float* conv18_bn;
static float* pool18_blob;

void conv_bn(float* ifm, float* ofm, float* weight, float* Mean, float* Variale, float* Bias, layer l)
{
    float* conv_blob = (float*)malloc(l.oc*l.oh*l.ow*sizeof(float));
    float* zeros = (float*)calloc(l.oc,sizeof(float));
    convolution_mm(ifm,conv_blob,weight,zeros,l);
    //convolution(ifm,conv_blob,weight,zeros,l);
    batchnorm(conv_blob, ofm, Mean, Variale, Bias, l);
    free(conv_blob);
    free(zeros);
}


int max(float *in,layer l)
{
    int index = 0;
    float tmp=in[0];
    for (int i = 1; i < l.oc; i++)
    {
        if (in[i] > tmp)
        {
            index = i;
            tmp = in[i];
        }
    }
    return index;
}

void Darknet_init()
{
    image_blob = (float*)malloc(DNet[image].oc*DNet[image].oh*DNet[image].ow*sizeof(float));

    conv1_blob = (float*)malloc(DNet[conv1].oc*DNet[conv1].oh*DNet[conv1].ow*sizeof(float));
    conv1_weight = (float*)malloc(DNet[conv1].ic*DNet[conv1].oc*DNet[conv1].k*DNet[conv1].k*sizeof(float));
    conv1_bn = (float*)malloc(DNet[conv1].oc*3*sizeof(float));

    pool1_blob = (float*)malloc(DNet[pool1].oc*DNet[pool1].oh*DNet[pool1].ow*sizeof(float));

    conv2_blob = (float*)malloc(DNet[conv2].oc*DNet[conv2].oh*DNet[conv2].ow*sizeof(float));
    conv2_weight = (float*)malloc(DNet[conv2].ic*DNet[conv2].oc*DNet[conv2].k*DNet[conv2].k*sizeof(float));
    conv2_bn = (float*)malloc(DNet[conv2].oc*3*sizeof(float));

    pool2_blob = (float*)malloc(DNet[pool2].oc*DNet[pool2].oh*DNet[pool2].ow*sizeof(float));

    conv3_blob = (float*)malloc(DNet[conv3].oc*DNet[conv3].oh*DNet[conv3].ow*sizeof(float));
    conv3_weight = (float*)malloc(DNet[conv3].ic*DNet[conv3].oc*DNet[conv3].k*DNet[conv3].k*sizeof(float));
    conv3_bn = (float*)malloc(DNet[conv3].oc*3*sizeof(float));

    conv4_blob = (float*)malloc(DNet[conv4].oc*DNet[conv4].oh*DNet[conv4].ow*sizeof(float));
    conv4_weight = (float*)malloc(DNet[conv4].ic*DNet[conv4].oc*DNet[conv4].k*DNet[conv4].k*sizeof(float));
    conv4_bn = (float*)malloc(DNet[conv4].oc*3*sizeof(float));

    conv5_blob = (float*)malloc(DNet[conv5].oc*DNet[conv5].oh*DNet[conv5].ow*sizeof(float));
    conv5_weight = (float*)malloc(DNet[conv5].ic*DNet[conv5].oc*DNet[conv5].k*DNet[conv5].k*sizeof(float));
    conv5_bn = (float*)malloc(DNet[conv5].oc*3*sizeof(float));

    pool5_blob = (float*)malloc(DNet[pool5].oc*DNet[pool5].oh*DNet[pool5].ow*sizeof(float));

    conv6_blob = (float*)malloc(DNet[conv6].oc*DNet[conv6].oh*DNet[conv6].ow*sizeof(float));
    conv6_weight = (float*)malloc(DNet[conv6].ic*DNet[conv6].oc*DNet[conv6].k*DNet[conv6].k*sizeof(float));
    conv6_bn = (float*)malloc(DNet[conv6].oc*3*sizeof(float));

    conv7_blob = (float*)malloc(DNet[conv7].oc*DNet[conv7].oh*DNet[conv7].ow*sizeof(float));
    conv7_weight = (float*)malloc(DNet[conv7].ic*DNet[conv7].oc*DNet[conv7].k*DNet[conv7].k*sizeof(float));
    conv7_bn = (float*)malloc(DNet[conv7].oc*3*sizeof(float));

    conv8_blob = (float*)malloc(DNet[conv8].oc*DNet[conv8].oh*DNet[conv8].ow*sizeof(float));
    conv8_weight = (float*)malloc(DNet[conv8].ic*DNet[conv8].oc*DNet[conv8].k*DNet[conv8].k*sizeof(float));
    conv8_bn = (float*)malloc(DNet[conv8].oc*3*sizeof(float));

    pool8_blob = (float*)malloc(DNet[pool8].oc*DNet[pool8].oh*DNet[pool8].ow*sizeof(float));

    conv9_blob = (float*)malloc(DNet[conv9].oc*DNet[conv9].oh*DNet[conv9].ow*sizeof(float));
    conv9_weight = (float*)malloc(DNet[conv9].ic*DNet[conv9].oc*DNet[conv9].k*DNet[conv9].k*sizeof(float));
    conv9_bn = (float*)malloc(DNet[conv9].oc*3*sizeof(float));

    conv10_blob = (float*)malloc(DNet[conv10].oc*DNet[conv10].oh*DNet[conv10].ow*sizeof(float));
    conv10_weight = (float*)malloc(DNet[conv10].ic*DNet[conv10].oc*DNet[conv10].k*DNet[conv10].k*sizeof(float));
    conv10_bn = (float*)malloc(DNet[conv10].oc*3*sizeof(float));

    conv11_blob = (float*)malloc(DNet[conv11].oc*DNet[conv11].oh*DNet[conv11].ow*sizeof(float));
    conv11_weight = (float*)malloc(DNet[conv11].ic*DNet[conv11].oc*DNet[conv11].k*DNet[conv11].k*sizeof(float));
    conv11_bn = (float*)malloc(DNet[conv11].oc*3*sizeof(float));

    conv12_blob = (float*)malloc(DNet[conv12].oc*DNet[conv12].oh*DNet[conv12].ow*sizeof(float));
    conv12_weight = (float*)malloc(DNet[conv12].ic*DNet[conv12].oc*DNet[conv12].k*DNet[conv12].k*sizeof(float));
    conv12_bn = (float*)malloc(DNet[conv12].oc*3*sizeof(float));

    conv13_blob = (float*)malloc(DNet[conv13].oc*DNet[conv13].oh*DNet[conv13].ow*sizeof(float));
    conv13_weight = (float*)malloc(DNet[conv13].ic*DNet[conv13].oc*DNet[conv13].k*DNet[conv13].k*sizeof(float));
    conv13_bn = (float*)malloc(DNet[conv13].oc*3*sizeof(float));

    pool13_blob = (float*)malloc(DNet[pool13].oc*DNet[pool13].oh*DNet[pool13].ow*sizeof(float));

    conv14_blob = (float*)malloc(DNet[conv14].oc*DNet[conv14].oh*DNet[conv14].ow*sizeof(float));
    conv14_weight = (float*)malloc(DNet[conv14].ic*DNet[conv14].oc*DNet[conv14].k*DNet[conv14].k*sizeof(float));
    conv14_bn = (float*)malloc(DNet[conv14].oc*3*sizeof(float));

    conv15_blob = (float*)malloc(DNet[conv15].oc*DNet[conv15].oh*DNet[conv15].ow*sizeof(float));
    conv15_weight = (float*)malloc(DNet[conv15].ic*DNet[conv15].oc*DNet[conv15].k*DNet[conv15].k*sizeof(float));
    conv15_bn = (float*)malloc(DNet[conv15].oc*3*sizeof(float));

    conv16_blob = (float*)malloc(DNet[conv16].oc*DNet[conv16].oh*DNet[conv16].ow*sizeof(float));
    conv16_weight = (float*)malloc(DNet[conv16].ic*DNet[conv16].oc*DNet[conv16].k*DNet[conv16].k*sizeof(float));
    conv16_bn = (float*)malloc(DNet[conv16].oc*3*sizeof(float));

    conv17_blob = (float*)malloc(DNet[conv17].oc*DNet[conv17].oh*DNet[conv17].ow*sizeof(float));
    conv17_weight = (float*)malloc(DNet[conv17].ic*DNet[conv17].oc*DNet[conv17].k*DNet[conv17].k*sizeof(float));
    conv17_bn = (float*)malloc(DNet[conv17].oc*3*sizeof(float));

    conv18_blob = (float*)malloc(DNet[conv18].oc*DNet[conv18].oh*DNet[conv18].ow*sizeof(float));
    conv18_weight = (float*)malloc(DNet[conv18].ic*DNet[conv18].oc*DNet[conv18].k*DNet[conv18].k*sizeof(float));
    conv18_bn = (float*)malloc(DNet[conv18].oc*3*sizeof(float));

    pool18_blob = (float*)malloc(DNet[pool18].oc*DNet[pool18].oh*DNet[pool18].ow*sizeof(float));
}

void Darknet_close()
{
    free(image_blob);

    free(conv1_blob);
    free(conv1_weight);
    free(conv1_bn);

    free(pool1_blob);
	
	free(conv2_blob);
    free(conv2_weight);
    free(conv2_bn);
	
    free(pool2_blob);

	free(conv3_blob);
    free(conv3_weight);
    free(conv3_bn);
	
	free(conv4_blob);
    free(conv4_weight);
    free(conv4_bn);
	
	free(conv5_blob);
    free(conv5_weight);
    free(conv5_bn);
	
    free(pool5_blob);

	free(conv6_blob);
    free(conv6_weight);
    free(conv6_bn);
	
	free(conv7_blob);
    free(conv7_weight);
    free(conv7_bn);
	
	free(conv8_blob);
    free(conv8_weight);
    free(conv8_bn);
	
    free(pool8_blob);

	free(conv9_blob);
    free(conv9_weight);
    free(conv9_bn);
	
	free(conv10_blob);
    free(conv10_weight);
    free(conv10_bn);
	
	free(conv11_blob);
    free(conv11_weight);
    free(conv11_bn);
	
	free(conv12_blob);
    free(conv12_weight);
    free(conv12_bn);
	
	free(conv13_blob);
    free(conv13_weight);
    free(conv13_bn);
	
    free(pool13_blob);

	free(conv14_blob);
    free(conv14_weight);
    free(conv14_bn);
	
	free(conv15_blob);
    free(conv15_weight);
    free(conv15_bn);
	
	free(conv16_blob);
    free(conv16_weight);
    free(conv16_bn);
	
	free(conv17_blob);
    free(conv17_weight);
    free(conv17_bn);
	
	free(conv18_blob);
    free(conv18_weight);
    free(conv18_bn);

    free(pool18_blob);
}

void load_DNet()
{
    load_weight(conv1_weight,DNet[conv1], Name);
    load_mean(conv1_bn,DNet[conv1], Name);
	
	load_weight(conv2_weight,DNet[conv2], Name);
    load_mean(conv2_bn,DNet[conv2], Name);
	
	load_weight(conv3_weight,DNet[conv3], Name);
    load_mean(conv3_bn,DNet[conv3], Name);
	
	load_weight(conv4_weight,DNet[conv4], Name);
    load_mean(conv4_bn,DNet[conv4], Name);
	
	load_weight(conv5_weight,DNet[conv5], Name);
    load_mean(conv5_bn,DNet[conv5], Name);
	
	load_weight(conv6_weight,DNet[conv6], Name);
    load_mean(conv6_bn,DNet[conv6], Name);
	
	load_weight(conv7_weight,DNet[conv7], Name);
    load_mean(conv7_bn,DNet[conv7], Name);
	
	load_weight(conv8_weight,DNet[conv8], Name);
    load_mean(conv8_bn,DNet[conv8], Name);
	
	load_weight(conv9_weight,DNet[conv9], Name);
    load_mean(conv9_bn,DNet[conv9], Name);
	
	load_weight(conv10_weight,DNet[conv10], Name);
    load_mean(conv10_bn,DNet[conv10], Name);
	
	load_weight(conv11_weight,DNet[conv11], Name);
    load_mean(conv11_bn,DNet[conv11], Name);
	
	load_weight(conv12_weight,DNet[conv12], Name);
    load_mean(conv12_bn,DNet[conv12], Name);
	
	load_weight(conv13_weight,DNet[conv13], Name);
    load_mean(conv13_bn,DNet[conv13], Name);
	
	load_weight(conv14_weight,DNet[conv14], Name);
    load_mean(conv14_bn,DNet[conv14], Name);
	
	load_weight(conv15_weight,DNet[conv15], Name);
    load_mean(conv15_bn,DNet[conv15], Name);
	
	load_weight(conv16_weight,DNet[conv16], Name);
    load_mean(conv16_bn,DNet[conv16], Name);
	
	load_weight(conv17_weight,DNet[conv17], Name);
    load_mean(conv17_bn,DNet[conv17], Name);
	
	load_weight(conv18_weight,DNet[conv18], Name);

}

int Darknet19(float* input)
{
    timeval start,end;
    gettimeofday(&start, NULL);
    conv_bn(input, conv1_blob, conv1_weight, conv1_bn, &conv1_bn[DNet[conv1].oc], &conv1_bn[2*DNet[conv1].oc], DNet[conv1]);
    maxpool(conv1_blob,pool1_blob,DNet[pool1]);
    conv_bn(pool1_blob, conv2_blob, conv2_weight, conv2_bn, &conv2_bn[DNet[conv2].oc], &conv2_bn[2*DNet[conv2].oc], DNet[conv2]);
    maxpool(conv2_blob,pool2_blob,DNet[pool2]);
    conv_bn(pool2_blob, conv3_blob, conv3_weight, conv3_bn, &conv3_bn[DNet[conv3].oc], &conv3_bn[2*DNet[conv3].oc], DNet[conv3]);
    conv_bn(conv3_blob, conv4_blob, conv4_weight, conv4_bn, &conv4_bn[DNet[conv4].oc], &conv4_bn[2*DNet[conv4].oc], DNet[conv4]);
    conv_bn(conv4_blob, conv5_blob, conv5_weight, conv5_bn, &conv5_bn[DNet[conv5].oc], &conv5_bn[2*DNet[conv5].oc], DNet[conv5]);
    maxpool(conv5_blob,pool5_blob,DNet[pool5]);
    conv_bn(pool5_blob, conv6_blob, conv6_weight, conv6_bn, &conv6_bn[DNet[conv6].oc], &conv6_bn[2*DNet[conv6].oc], DNet[conv6]);
    conv_bn(conv6_blob, conv7_blob, conv7_weight, conv7_bn, &conv7_bn[DNet[conv7].oc], &conv7_bn[2*DNet[conv7].oc], DNet[conv7]);
    conv_bn(conv7_blob, conv8_blob, conv8_weight, conv8_bn, &conv8_bn[DNet[conv8].oc], &conv8_bn[2*DNet[conv8].oc], DNet[conv8]);
    maxpool(conv8_blob,pool8_blob,DNet[pool8]);
    conv_bn(pool8_blob, conv9_blob, conv9_weight, conv9_bn, &conv9_bn[DNet[conv9].oc], &conv9_bn[2*DNet[conv9].oc], DNet[conv9]);
    conv_bn(conv9_blob, conv10_blob, conv10_weight, conv10_bn, &conv10_bn[DNet[conv10].oc], &conv10_bn[2*DNet[conv10].oc], DNet[conv10]);
    conv_bn(conv10_blob, conv11_blob, conv11_weight, conv11_bn, &conv11_bn[DNet[conv11].oc], &conv11_bn[2*DNet[conv11].oc], DNet[conv11]);
    conv_bn(conv11_blob, conv12_blob, conv12_weight, conv12_bn, &conv12_bn[DNet[conv12].oc], &conv12_bn[2*DNet[conv12].oc], DNet[conv12]);
    conv_bn(conv12_blob, conv13_blob, conv13_weight, conv13_bn, &conv13_bn[DNet[conv13].oc], &conv13_bn[2*DNet[conv13].oc], DNet[conv13]);
    maxpool(conv13_blob,pool13_blob,DNet[pool13]);
    conv_bn(pool13_blob, conv14_blob, conv14_weight, conv14_bn, &conv14_bn[DNet[conv14].oc], &conv14_bn[2*DNet[conv14].oc], DNet[conv14]);
    conv_bn(conv14_blob, conv15_blob, conv15_weight, conv15_bn, &conv15_bn[DNet[conv15].oc], &conv15_bn[2*DNet[conv15].oc], DNet[conv15]);
    conv_bn(conv15_blob, conv16_blob, conv16_weight, conv16_bn, &conv16_bn[DNet[conv16].oc], &conv16_bn[2*DNet[conv16].oc], DNet[conv16]);
    conv_bn(conv16_blob, conv17_blob, conv17_weight, conv17_bn, &conv17_bn[DNet[conv17].oc], &conv17_bn[2*DNet[conv17].oc], DNet[conv17]);
    float* zeros = (float*)calloc(DNet[conv18].oc,sizeof(float));
    convolution(conv17_blob,conv18_blob,conv18_weight,zeros,DNet[conv18]);
    avgpool(conv18_blob,pool18_blob,DNet[pool18]);
    int label = max(pool18_blob, DNet[pool18]);
    gettimeofday(&end, NULL);
    long us = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    printf("Darknet19 took %lu us\n", us);
    return label;
}
