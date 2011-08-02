/*
 * AamMatchFunction.h
 *
 *  Created on: Jul 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCVCUDA_AAMMATCHFUNCTION_H_
#define GAPPUTILSCVCUDA_AAMMATCHFUNCTION_H_

#include <vector>
#include <culib/ICudaImage.h>
#include <culib/similarity.h>
#include <thrust/device_vector.h>

namespace gapputils {

namespace cv {

namespace cuda {

struct AamMatchStatus {
  std::vector<float> shapeParameters;
  thrust::device_vector<float> d_shapeParameters;
  thrust::device_vector<float> d_textureParameters;
  thrust::device_vector<float> d_appearanceParameters;

  thrust::device_vector<float> d_shapeFeatures;
  thrust::device_vector<float> d_textureFeatures;
  thrust::device_vector<float> d_appearanceFeatures;

  AamMatchStatus(int spCount, int tpCount, int apCount, int sfCount, int tfCount, int afCount);
};

double aamMatchFunction(const std::vector<double>& parameter, int spCount, int tpCount, int apCount, int width, int height,
    int columnCount, int rowCount, float* d_shapeMatrix, float* d_textureMatrix, float* d_appearanceMatrix,
    float* d_meanShape, float* d_meanTexture,
    AamMatchStatus& status,
    culib::ICudaImage* inputImage, culib::ICudaImage* warpedImage,
    bool inReferenceFrame, culib::SimilarityConfig& config, bool useMi);

}

}

}



#endif /* GAPPUTILSCVCUDA_AAMMATCHFUNCTION_H_ */
