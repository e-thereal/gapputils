#pragma once
#ifndef GAPPUTILS_CV_CUDA_AGGREGATE_H_
#define GAPPUTILS_CV_CUDA_AGGREGATE_H_

#include <culib/ICudaImage.h>

namespace gapputils {

namespace cv {

namespace cuda {

void average(culib::ICudaImage* input, culib::ICudaImage* output);
void sum(culib::ICudaImage* input, culib::ICudaImage* output);

}

}

}

#endif /* GAPPUTILS_CV_CUDA_AGGREGATE_H_ */
