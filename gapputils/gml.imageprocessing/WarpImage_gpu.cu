/*
 * WarpImage_gpu.cu
 *
 *  Created on: Nov 7, 2014
 *      Author: tombr
 */

#include "WarpImage.h"

#include <tbblas/transform/warp.hpp>

namespace gml {

namespace imageprocessing {

WarpImageChecker::WarpImageChecker() {
  WarpImage test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Input, test);
  CHECK_MEMORY_LAYOUT2(Deformation, test);
  CHECK_MEMORY_LAYOUT2(VoxelSize, test);
  CHECK_MEMORY_LAYOUT2(Output, test);
}

void WarpImage::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::transform;

  typedef float value_t;
  typedef tensor<value_t, 3, true> volume_t;
  typedef tensor<value_t, 4, true> tensor_t;

  image_t& input = *getInput();

  // TODO: Test 4D warp by reshaping a 3D image into a 4D tensor with one channel

  volume_t invol(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
  thrust::copy(input.begin(), input.end(), invol.begin());

  tensor_t deformation = *getDeformation();
  volume_t outvol = warp(invol, deformation, seq(_VoxelSize[0], _VoxelSize[1], _VoxelSize[2]));

  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));
  thrust::copy(outvol.begin(), outvol.end(), output->begin());

  newState->setOutput(output);
}

}

}
