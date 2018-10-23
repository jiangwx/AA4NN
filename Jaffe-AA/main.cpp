#include "Darknet19.h"
#define TEST 10
char synset[30][35] = {
        {"A-line-loose-dress\0"},
        {"Canvas-shoes\0"},
        {"Long-Jeans\0"},
        {"beidaiqun\0"},
        {"booties-short-boots\0"},
        {"cardigans\0"},
        {"flat-sandals\0"},
        {"fur\0"},
        {"heel-sandals\0"},
        {"hoodies\0"},
        {"leather-loafer-shoes\0"},
        {"leisure-dress\0"},
        {"leisure-pants\0"},
        {"30longsleeve-dress-for-fall\0"},
        {"mid-long-boots\0"},
        {"outdoor-xiaobao\0"},
        {"outerwear-vest\0"},
        {"pullover-tops\0"},
        {"qipao\0"},
        {"shirt-dress\0"},
        {"shorts\0"},
        {"shoulder-handbag\0"},
        {"slim-bottom-skirt\0"},
        {"slim-pants-leggings\0"},
        {"tank-strap-tops\0"},
        {"trenchcoat\0"},
        {"volumn-skirt\0"},
        {"waisted-dress\0"},
        {"wallets\0"},
        {"women-Sports-jackets\0"},
};
/*
int main(int argc, char *argv[])
{
	FILE *bb;
    char* image_path = argv[1];
    char* mean_path = argv[2];
    float *image = (float *)malloc(sizeof(float)*img_w*img_h*3);
    float *mean = (float *)malloc(sizeof(float)*img_w*img_h*3);
    int label;

    load_mean_image(mean_path,mean);
    load_image(image_path, image, mean);

    Darknet_init();
    load_DNet();
    label = Darknet19(image);

    printf("%s %s\n", image_path, synset[label]);

    Darknet_close();
    return 0;
}*/

int main(int argc, char **argv)
{
    int label;
    FILE *bb;
    char* image_folder = argv[1];
    char mean_path[50];
    char image_path[50];

    float *image = (float *)malloc(sizeof(float)*img_w*img_h*3);
    float *mean = (float *)malloc(sizeof(float)*img_w*img_h*3);

    sprintf(mean_path, "%s/mean.jpg", image_folder);
    load_mean_image(mean_path,mean);
    Darknet_init();
    load_DNet();

    for(int i=1;i<TEST+1;i++) {
        sprintf(image_path, "%s/%08d.jpg", image_folder, i);
        load_image(image_path, image, mean);
        label = Darknet19(image);
        printf("%08d.jpg %s\n", i, synset[label]);
    }
    Darknet_close();
}
/*
int main(int argc, char **argv)
{
    int label;
    FILE *bb;
    char* image_folder = argv[1];
    char image_path[50];
    float *blob = (float*)malloc(sizeof(float)*img_w*img_h*3);

    Darknet_init();
    load_DNet();

    for(int i=1;i<TEST+1;i++) {
        sprintf(image_path, "%s/%08d.bb", image_folder, i);
        bb = fopen(image_path, "rb");
        fread(blob,sizeof(float),img_w*img_h*3,bb);
        label = Darknet19(blob);
        printf("%08d.jpg %s\n", i, synset[label]);
    }
    fclose(bb);
}
*/