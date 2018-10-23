#ifndef DARKNET19_H
#define DARKNET19_H
#include "CNN.h"

void Darknet_init();
void Darknet_close();
void load_DNet();
int Darknet19(float* input);
int AA_Darknet19(float* input, char* image_path);
#endif //JAFFE_AA_DARKNET19_H
