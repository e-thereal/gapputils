/*
 * Compare_gpu.cu
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#include "Compare.h"

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>
#include <tbblas/conv.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>

namespace gml {

namespace imageprocessing {

CompareChecker::CompareChecker() {
  Compare compare;
  compare.initializeClass();
  CHECK_MEMORY_LAYOUT2(Image1, compare);
  CHECK_MEMORY_LAYOUT2(Image2, compare);
  CHECK_MEMORY_LAYOUT2(Measure, compare);
  CHECK_MEMORY_LAYOUT2(Parameters, compare);
  CHECK_MEMORY_LAYOUT2(Value, compare);

  SsimParameters ssim;
  ssim.initializeClass();
  CHECK_MEMORY_LAYOUT2(WindowWidth, ssim);
  CHECK_MEMORY_LAYOUT2(WindowHeight, ssim);
  CHECK_MEMORY_LAYOUT2(WindowDepth, ssim);
}

void Compare::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor<float, 3, true> tensor_t;

  Logbook& dlog = getLogbook();

  image_t& input1 = *getImage1();
  image_t& input2 = *getImage2();

  tensor_t img1(input1.getSize()[0], input1.getSize()[1], input1.getSize()[2]);
  tensor_t img2(input2.getSize()[0], input2.getSize()[1], input2.getSize()[2]);

  thrust::copy(input1.begin(), input1.end(), img1.begin());
  thrust::copy(input2.begin(), input2.end(), img2.begin());

  switch (getMeasure()) {
  case SimilarityMeasure::MSE:
    img1 = (img1 - img2) * (img1 - img2);
    newState->setValue(sum(img1) / img1.count());
    return;

  case SimilarityMeasure::SSIM:
    {
      boost::shared_ptr<SsimParameters> params = boost::dynamic_pointer_cast<SsimParameters>(getParameters());
      if (params) {
        tensor_t averagePattern = ones<float>(params->getWindowWidth(), params->getWindowHeight(), params->getWindowDepth());
        averagePattern = averagePattern / averagePattern.count();

        // mu1 = E[X], mu2 = E[Y]
        tensor_t mu1 = conv(img1, averagePattern), mu2 = conv(img2, averagePattern);
        tensor_t mu1sq = mu1 * mu1, mu2sq = mu2 * mu2;

        // COV(X, Y) = E[XY] - E[X]E[Y]
        tensor_t img12 = img1 * img2;
        tensor_t sig12 = conv(img12, averagePattern);
        sig12 = sig12 - mu1 * mu2;

        // VAR(X) = E[X^2] - (E[X])^2
        tensor_t img1sq = img1 * img1, img2sq = img2 * img2;
        tensor_t var1 = conv(img1sq, averagePattern), var2 = conv(img2sq, averagePattern);
        var1 = var1 - mu1sq;
        var2 = var2 = mu2sq;

        const float c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
        tensor_t ssim = (2.f * mu1 * mu2 + c1) * (2 * sig12 + c2) /
            ((mu1sq + mu2sq + c1) * (var1 + var2 + c2));
        newState->setValue(sum(ssim) / ssim.count());
      }
    }
    return;

  default:
    dlog(Severity::Warning) << "Unsupported measure '" << getMeasure() << "'. Aborting!";
    return;
  }
}

}

}
