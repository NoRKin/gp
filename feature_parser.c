#include <stdlib.h>
#include "feature_parser.h"

float **feature_fromFile(FILE *fp, int nbSamples, int nbFeatures) {
    float **samples = malloc(nbSamples * sizeof(float *));
    int i = 0;
    int column = 0;

    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;
    linelen = getline(&line, &linecap, fp); // discard first line
    while (i < nbSamples && (linelen = getline(&line, &linecap, fp)) > 0) {
        samples[i] = malloc((nbFeatures + 1) * sizeof(float));
        int j = 0;
        int column = 0;
        char *parsedLine = line;
        // fprintf(stdout, "SAMPLE (");
        // Skip 3 first colums
        for (column = 0; column < 3; column++) {
          while (*parsedLine != ',') {
            ++parsedLine;
          }
          ++parsedLine;
        }

        while (j < nbFeatures) {
            samples[i][j] = atof(parsedLine);
            while (*parsedLine != ',' && *parsedLine != '\n') {
                ++parsedLine;
            }
            ++parsedLine;
            // fprintf(stdout, "%f,", samples[i][j]);
            ++j;
        }
        samples[i][j] = atof(parsedLine);
        // fprintf(stdout, ") = %f\n", samples[i][j]);
        ++i;
    }

    free(line);

    return samples;
}


float *flatten_features(const float **featuresPtr, int dataset_size, int feature_count) {
  int idx = 0;
  float *flattened = (float *)malloc(sizeof(float) * dataset_size * feature_count);

  for (int i = 0; i < dataset_size; i++) {
    for (int x = 0; x < feature_count; x++) {
      flattened[idx] = featuresPtr[i][x];
      idx++;
    }
  }

  return flattened;
}
