#ifndef GAPPUTILS_CV_CUDA_IMAGEMATRIX_HPP
#define GAPPUTILS_CV_CUDA_IMAGEMATRIX_HPP

#include "../culib/ICudaImage.h"

namespace gapputils {

namespace cv {

namespace cuda {

void createImageMatrix(culib::ICudaImage& input, culib::ICudaImage& imageMatrix);

}

}

}

#endif /* GAPPUTILS_CV_CUDA_IMAGEMATRIX_HPP */