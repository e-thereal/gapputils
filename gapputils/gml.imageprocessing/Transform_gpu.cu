/*
 * Transform_gpu.cu
 *
 *  Created on: Jul 23, 2015
 *      Author: tombr
 */

#include "Transform.h"

#include <tbblas/tensor.hpp>
#include <tbblas/imgproc/transform.hpp>

namespace gml {

namespace imageprocessing {

void Transform::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::imgproc;

  image_t& input = *getInput();

  tensor<float, 3, true> in(input.getSize()[0], input.getSize()[1], input.getSize()[2]), out;
  thrust::copy(input.begin(), input.end(), in.begin());

  out = transform(in, *getTransform());

  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));
  thrust::copy(out.begin(), out.end(), output->begin());
  newState->setOutput(output);
}

}

}
