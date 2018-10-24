#include "CNN.h"
#define GNet_depth 85

layer GNet[GNet_depth] = {
        { defualt_, "image",      1248,384,3,1248,384,3,0,0,0},
        { conv_,    "conv177",    1248,384,3,624,192,64,7,2,3 },
        { maxpool_, "pool133",    624,192,64,312,96,64,3,2,0 },
        { lrn_,     "pool1norm",  312,96,64,312,96,64,0,0,0 },
        { conv_,    "conv233re",  312,96,64,312,96,64,1,1,0 },
        { conv_,    "conv233",    312,96,64,312,96,192,3,1,1 },
        { conv_,    "conv2norm",  312,96,192,312,96,192,0,0,0 },
        { maxpool_, "pool233",    312,96,192,156,48,192,3,2,0 },

        { conv_,    "incep3a11",  156,48,192,156,48,64,1,1,0 },
        { conv_,    "incep3a33re",156,48,192,156,48,96,1,1,0 },
        { conv_,    "incep3a33",  156,48,96,156,48,128,3,1,1 },
        { conv_,    "incep3a55re",156,48,192,156,48,16,1,1,0 },
        { conv_,    "incep3a55",  156,48,16,156,48,32,5,1,2 },
        { maxpool_, "incep3apo",  156,48,192,156,48,192,3,1,1 },
        { conv_,    "incep3app",  156,48,192,156,48,32,1,1,0 },
        { concat_,  "incep3a",    156,48,192,156,48,256,0,0,0 },

        { conv_,    "incep3b11",  156,48,256,156,48,128,1,1,0 },
        { conv_,    "incep4b33re",156,48,256,156,48,128,1,1,0 },
        { conv_,    "incep3b33",  156,48,128,156,48,192,3,1,1 },
        { conv_,    "incep3b55re",156,48,256,156,48,32,1,1,0 },
        { conv_,    "incep3b55",  156,48,32,156,48,96,5,1,2 },
        { maxpool_, "incep3bpo",  156,48,256,156,48,256,3,1,1 },
        { conv_,    "incep3bpp",  156,48,256,156,48,64,1,1,0 },
        { concat_,  "incep3b",    156,48,256,156,48,480,0,0,0 },

        { maxpool_, "pool333",    156,48,480,78,24,480,3,2,0 },

        { conv_,    "incep4a11",  78,24,480,78,24,192,1,1,0 },
        { conv_,    "incep4a33re",78,24,480,78,24,96,1,1,0 },
        { conv_,    "incep4a33",  78,24,96,78,24,208,3,1,1 },
        { conv_,    "incep4a55re",78,24,480,78,24,16,1,1,0 },
        { conv_,    "incep4a55",  78,24,16,78,24,48,5,1,2 },
        { maxpool_, "incep4apo",  78,24,480,78,24,480,3,1,1 },
        { conv_,    "incep4app",  78,24,480,78,24,64,1,1,0 },
        { concat_,  "incep4a",    78,24,480,78,24,512,0,0,0 },


        { conv_,"incep4b11",  78,24,512,78,24,160,1,1,0 },
        { conv_,"incep4b33re",78,24,512,78,24,112,1,1,0 },
        { conv_,"incep4b33",  78,24,112,78,24,224,3,1,1 },
        { conv_,"incep4b55re",78,24,512,78,24,24,1,1,0 },
        { conv_,"incep4b55",  78,24,24,78,24,64,5,1,2 },
        { maxpool_,"incep4bpo",  78,24,512,78,24,512,3,1,1 },
        { conv_,"incep4bpp",  78,24,512,78,24,64,1,1,0 },
        { concat_,"incep4b",    78,24,512,78,24,512,0,0,0 },

        { conv_,"incep4c11",  78,24,512,78,24,128,1,1,0 },
        { conv_,"incep4c33re",78,24,512,78,24,128,1,1,0 },
        { conv_,"incep4c33",  78,24,128,78,24,256,3,1,1 },
        { conv_,"incep4c55re",78,24,512,78,24,24,1,1,0 },
        { conv_,"incep4c55",  78,24,24,78,24,64,5,1,2 },
        { maxpool_,"incep4cpo",  78,24,512,78,24,512,3,1,1 },
        { conv_,"incep4cpp",  78,24,512,78,24,64,1,1,0 },
        { concat_,"incep4c",    78,24,512,78,24,512,0,0,0 },

        { conv_,"incep4d11",  78,24,512,78,24,112,1,1,0 },
        { conv_,"incep4d33re",78,24,512,78,24,144,1,1,0 },
        { conv_,"incep4d33",  78,24,144,78,24,288,3,1,1 },
        { conv_,"incep4d55re",78,24,512,78,24,32,1,1,0 },
        { conv_,"incep4d55",  78,24,32,78,24,64,5,1,2 },
        { maxpool_,"incep4dpo",  78,24,512,78,24,512,3,1,1 },
        { conv_,"incep4dpp",  78,24,512,78,24,64,1,1,0 },
        { concat_,"incep4d",    78,24,512,78,24,528,0,0,0 },

        { conv_,"incep4e11",  78,24,528,78,24,256,1,1,0 },
        { conv_,"incep4e33re",78,24,528,78,24,160,1,1,0 },
        { conv_,"incep4e33",  78,24,160,78,24,320,3,1,1 },
        { conv_,"incep4e55re",78,24,528,78,24,32,1,1,0 },
        { conv_,"incep4e55",  78,24,32,78,24,128,5,1,2 },
        { maxpool_,"incep4epo",  78,24,528,78,24,528,3,1,1 },
        { conv_,"incep4epp",  78,24,528,78,24,128,1,1,0 },
        { concat_,"incep4e",    78,24,528,78,24,832,0,0,0 },

        { conv_,"incep5a11",  78,24,832,78,24,256,1,1,0 },
        { conv_,"incep5a33re",78,24,832,78,24,160,1,1,0 },
        { conv_,"incep5a33",  78,24,160,78,24,320,3,1,1 },
        { conv_,"incep5a55re",78,24,832,78,24,32,1,1,0 },
        { conv_,"incep5a55",  78,24,32,78,24,128,5,1,2 },
        { maxpool_,"incep5apo",  78,24,832,78,24,832,3,1,1 },
        {conv_, "incep5app",  78,24,832,78,24,128,1,1,0 },
        { concat_,"incep5a",    78,24,832,78,24,832,0,0,0 },

        { conv_,"incep5b11",   78,24,832,78,24,384,1,1,0 },
        { conv_,"incep5b33re", 78,24,832,78,24,192,1,1,0 },
        { conv_,"incep5b33",   78,24,192,78,24,384,3,1,1 },
        { conv_,"incep5b55re", 78,24,832,78,24,48,1,1,0 },
        { conv_,"incep5b55",   78,24,48,78,24,128,5,1,2 },
        { maxpool_,"incep5bpo",   78,24,832,78,24,832,3,1,1 },
        { conv_,"incep5bpp",   78,24,832,78,24,128,1,1,0 },
        { concat_,"incep5b",     78,24,832,78,24,1024,0,0,0 },

        { defualt_,"pool5drops",   78,24,1024,78,24,1,1 },
        { conv_,"cvgclassifier",78,24,1024,78,24,1,1,1,0 },
        { sigmoid_,"cvgsig",       78,24,1,78,24,1,0,0,0 },
        { conv_,"bboxes",       78,24,1024,78,24,4,1,1,0 },
};

enum GNet_idx {
    image,conv177, pool133, pool1norm, conv233re, conv233, conv2norm, pool233,
    incep3a11, incep3a33re, incep3a33, incep3a55re, incep3a55, incep3apo, incep3app, incep3a,
    incep3b11, incep3b33re, incep3b33, incep3b55re, incep3b55, incep3bpo, incep3bpp, incep3b,
    pool333,
    incep4a11, incep4a33re, incep4a33, incep4a55re, incep4a55, incep4apo, incep4app, incep4a,
    incep4b11, incep4b33re, incep4b33, incep4b55re, incep4b55, incep4bpo, incep4bpp, incep4b,
    incep4c11, incep4c33re, incep4c33, incep4c55re, incep4c55, incep4cpo, incep4cpp, incep4c,
    incep4d11, incep4d33re, incep4d33, incep4d55re, incep4d55, incep4dpo, incep4dpp, incep4d,
    incep4e11, incep4e33re, incep4e33, incep4e55re, incep4e55, incep4epo, incep4epp, incep4e,
    incep5a11, incep5a33re, incep5a33, incep5a55re, incep5a55, incep5apo, incep5app, incep5a,
    incep5b11, incep5b33re, incep5b33, incep5b55re, incep5b55, incep5bpo, incep5bpp, incep5b,
    pool5drops, cvgclassifier, cvgsig, bboxes
};
/*
static char Name[]="Googlenet";

static float* image_blob;
static float* conv177_output;
static float* conv177relu_output;
static AAF* conv177relu_output_AA;
static float* conv177_weight;
static float* conv177_bias;
static float* pool133_output;
static AAF* pool133_output_AA;
static float* pool1norm_output;
static AAF* pool1norm_output_AA;
static float* conv233re_output;
static AAF* conv233re_output_AA;
static float* conv233rerelu_output;
static AAF* conv233rerelu_output_AA;
static float* conv233re_weight;
static float* conv233re_bias;
static float* conv233_output;
static AAF* conv233_output_AA;
static float* conv233relu_output;
static AAF* conv233relu_output_AA;
static float* conv233_weight;
static float* conv233_bias;
static float* conv2norm_output;
static AAF* conv2norm_output_AA;
static float* pool233_output;
static AAF* pool233_output_AA;

void init_gnet()
{
    int ptr_cnt = 0;
    for(int i=0;i<GNet_depth;i++)
    {
        switch (GNet[i].type)
        {
            case io_:  ptr_cnt+=1;break;
            case conv_:ptr_cnt+=3;break;
            case maxpool_:ptr_cnt+=1;break;
            case avgpool_:ptr_cnt+=1;break;
            case lrn_:ptr_cnt+=1;break;
            case sigmoid_:ptr_cnt+=1;break;
            case innerproduct_:ptr_cnt+=3;break;
            case batchnorm_:ptr_cnt+=1;break;
            case relu_:ptr_cnt+=1;break;
            case concat_:ptr_cnt+=1;break;
            case defualt_:ptr_cnt+=1;break;
        }
    }


    image_blob = (float*)calloc(GNet[image].oc*GNet[image].oh*GNet[image].ow,sizeof(float));

    conv177_output = (float*)calloc(GNet[conv177].oc*GNet[conv177].oh*GNet[conv177].ow,sizeof(float));
    conv177relu_output = (float*)calloc(GNet[conv177].oc*GNet[conv177].oh*GNet[conv177].ow,sizeof(float));
    conv177relu_output_AA = new AAF[GNet[conv177].oc*GNet[conv177].oh*GNet[conv177].ow];
    conv177_weight = (float*)calloc(GNet[conv177].ic*GNet[conv177].oc*GNet[conv177].k*GNet[conv177].k,sizeof(float));
    conv177_bias = (float*)calloc(GNet[conv177].oc,sizeof(float));

    pool133_output = (float*)calloc(GNet[pool133].oc*GNet[pool133].oh*GNet[pool133].ow,sizeof(float));
    pool133_output_AA = new AAF[GNet[pool133].oc*GNet[pool133].oh*GNet[pool133].ow];
    pool1norm_output = (float*)calloc(GNet[pool1norm].oc*GNet[pool1norm].oh*GNet[pool1norm].ow,sizeof(float));
    pool1norm_output_AA = new AAF[GNet[pool1norm].oc*GNet[pool1norm].oh*GNet[pool1norm].ow];

    conv233re_output = (float*)calloc(GNet[conv233re].oc*GNet[conv233re].oh*GNet[conv233re].ow,sizeof(float));
    conv233re_output_AA = new AAF[GNet[conv233re].oc*GNet[conv233re].oh*GNet[conv233re].ow];
    conv233rerelu_output = (float*)calloc(GNet[conv233re].oc*GNet[conv233re].oh*GNet[conv233re].ow,sizeof(float));
    conv233rerelu_output_AA = new AAF[GNet[conv233re].oc*GNet[conv233re].oh*GNet[conv233re].ow];
    conv233re_weight = (float*)calloc(GNet[conv233re].ic*GNet[conv233re].oc*GNet[conv233re].k*GNet[conv233re].k,sizeof(float));
    conv233re_bias = (float*)calloc(GNet[conv233re].oc,sizeof(float));

    conv233_output = (float*)calloc(GNet[conv233].oc*GNet[conv233].oh*GNet[conv233].ow,sizeof(float));
    conv233_output_AA = new AAF[GNet[conv233].oc*GNet[conv233].oh*GNet[conv233].ow];
    conv233relu_output = (float*)calloc(GNet[conv233].oc*GNet[conv233].oh*GNet[conv233].ow,sizeof(float));
    conv233relu_output_AA = new AAF[GNet[conv233].oc*GNet[conv233].oh*GNet[conv233].ow];
    conv233_weight = (float*)calloc(GNet[conv233].ic*GNet[conv233].oc*GNet[conv233].k*GNet[conv233].k,sizeof(float));
    conv233_bias = (float*)calloc(GNet[conv233].oc,sizeof(float));

    conv2norm_output = (float*)calloc(GNet[conv2norm].oc*GNet[conv2norm].oh*GNet[conv2norm].ow,sizeof(float));
    conv2norm_output_AA = new AAF[GNet[conv2norm].oc*GNet[conv2norm].oh*GNet[conv2norm].ow];

    pool233_output = (float*)calloc(GNet[pool233].oc*GNet[pool233].oh*GNet[pool233].ow,sizeof(float));
    pool233_output_AA = new AAF[GNet[pool233].oc*GNet[pool233].oh*GNet[pool233].ow];
}

void load_gnet()
{
    load_weight(conv177_weight,GNet[conv177], Name);
    load_bias(conv177_bias,GNet[conv177], Name);

    load_weight(conv233re_weight,GNet[conv233re], Name);
    load_bias(conv233re_bias,GNet[conv233re], Name);

    load_weight(conv233_weight,GNet[conv233], Name);
    load_bias(conv233_bias,GNet[conv233], Name);
}

void AA_googlenet()
{
    timeval start,end;
    init_gnet();
    std::cout << "init googlenet" << std::endl;
    load_gnet();

    //load_fm(image_blob,GNet[image]);
    //convolution(image_blob,conv177_output,conv177_weight,conv177_bias,GNet[conv177]);
    //relu(conv177_output,conv177relu_output,GNet[conv177]);
    //check_fm(conv177relu_output,GNet[conv177]);

    load_fm(conv177relu_output,GNet[conv177],Name);
    AA_add(conv177relu_output,conv177relu_output_AA,GNet[conv177]);
    AA_maxpool(conv177relu_output_AA,pool133_output_AA,GNet[pool133]);
    std::cout << "pool133 over" << std::endl;
    AA_check_fm(pool133_output_AA,GNet[pool133],Name);
    AA_save_fm(pool133_output_AA,GNet[pool133],Name);

    //load_fm(pool133_output,GNet[pool133]);
    //AA_add(pool133_output,pool133_output_AA,GNet[pool133]);
    AA_lrn(pool133_output_AA,pool1norm_output_AA,5,0.0001,0.75,GNet[pool1norm]);
    std::cout << "pool1norm over" << std::endl;
    //AA_check_fm(pool1norm_output_AA,GNet[pool1norm]);

    load_fm(pool1norm_output,GNet[pool1norm],Name);
    AA_add(pool1norm_output,pool1norm_output_AA,GNet[pool1norm]);
    AA_convolution(pool1norm_output_AA,conv233re_output_AA,conv233re_weight,conv233re_bias,GNet[conv233re]);
    std::cout << "conv233re over" << std::endl;
    AA_relu(conv233re_output_AA,conv233rerelu_output_AA,GNet[conv233re]);
    std::cout << "conv233rerelu over" << std::endl;
    //AA_check_fm(conv233rerelu_output_AA,GNet[conv233re],Name);
    //AA_save_fm(conv233rerelu_output_AA,GNet[conv233re],Name);

    //load_fm(conv233rerelu_output,GNet[conv233re]);
    //AA_add(conv233rerelu_output,conv233rerelu_output_AA,GNet[conv233re]);
    AA_convolution(conv233rerelu_output_AA,conv233_output_AA,conv233_weight,conv233_bias,GNet[conv233]);
    std::cout << "conv233 over" << std::endl;
    AA_relu(conv233_output_AA,conv233relu_output_AA,GNet[conv233]);
    std::cout << "conv233relu over" << std::endl;
    //AA_check_fm(conv233relu_output_AA,GNet[conv233]);


    //load_fm(conv233relu_output,GNet[conv233]);
    //AA_add(conv233relu_output,conv233relu_output_AA,GNet[conv233]);
    AA_lrn(conv233relu_output_AA,conv2norm_output_AA,5,0.0001,0.75,GNet[conv2norm]);
    AA_maxpool(conv2norm_output_AA,pool233_output_AA,GNet[pool233]);
    AA_check_fm(pool233_output_AA,GNet[pool233],Name);
    AA_save_fm(pool233_output_AA,GNet[pool233],Name);
}
*/