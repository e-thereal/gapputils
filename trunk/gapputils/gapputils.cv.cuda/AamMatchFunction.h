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

namespace gapputils {

namespace cv {

namespace cuda {

double aamMatchFunction(const std::vector<double>& parameter, int spCount, int tpCount, int apCount, int width, int height,
    int columnCount, int rowCount, float* shapeMatrix, float* textureMatrix, float* appearanceMatrix,
    float* meanShape, float* meanTexture,
    culib::ICudaImage* inputImage, culib::ICudaImage* warpedImage,
    bool inReferenceFrame, culib::SimilarityConfig& config, bool useMi);

}

}

}



#endif /* GAPPUTILSCVCUDA_AAMMATCHFUNCTION_H_ */
