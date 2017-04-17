#ifndef __FEATURE_PARSER_H__
# define __FEATURE_PARSER_H__

#include <stdio.h>

float **feature_fromFile(FILE *featureFile, int nbSamples, int nbFeatures);
float *flatten_features(const float **featuresPtr, int dataset_size, int feature_count);

#endif //__FEATURE_PARSER_H__
