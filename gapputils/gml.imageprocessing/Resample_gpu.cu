/*
 * Resample_gpu.cu
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#include "Resample.h"

#include <tbblas/tensor.hpp>
#include <tbblas/imgproc/transform.hpp>

namespace gml {

namespace imageprocessing {

ResampleChecker::ResampleChecker () {
  Resample test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Input, test);
  CHECK_MEMORY_LAYOUT2(Size, test);
  CHECK_MEMORY_LAYOUT2(PixelSize, test);
  CHECK_MEMORY_LAYOUT2(Output, test);
}

void Resample::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::imgproc;

  typedef float value_t;
  typedef tensor<value_t, 3, true> volume_t;

  image_t& input = *getInput();
  boost::shared_ptr<image_t> output(new image_t(_Size[0], _Size[1], _Size[2], _PixelSize[0], _PixelSize[1], _PixelSize[2]));

  volume_t in(input.getSize()[0], input.getSize()[1], input.getSize()[2]), out;
  thrust::copy(input.begin(), input.end(), in.begin());

  float inXVoxelSize = float(input.getPixelSize()[0]) * 0.001f;
  float inYVoxelSize = float(input.getPixelSize()[1]) * 0.001f;
  float inZVoxelSize = float(input.getPixelSize()[2]) * 0.001f;

  float outXVoxelSize = float(output->getPixelSize()[0]) * 0.001f;
  float outYVoxelSize = float(output->getPixelSize()[1]) * 0.001f;
  float outZVoxelSize = float(output->getPixelSize()[2]) * 0.001f;

  // Find centers (-0.5, because we rotate around the center of a voxel)
  value_t infxc = value_t(input.getSize()[0]) / 2.0-0.5;
  value_t infyc = value_t(input.getSize()[1]) / 2.0-0.5;
  value_t infzc = value_t(input.getSize()[2]) / 2.0-0.5;

  value_t outfxc = value_t(output->getSize()[0]) / 2.0-0.5;
  value_t outfyc = value_t(output->getSize()[1]) / 2.0-0.5;
  value_t outfzc = value_t(output->getSize()[2]) / 2.0-0.5;

  fmatrix4 moveCenterToOrigin = make_fmatrix4_translation(-outfxc, -outfyc, -outfzc);
  fmatrix4 moveBack = make_fmatrix4_translation(infxc, infyc, infzc);

  // Transfer coordinates the physical space
  fmatrix4 applyDimension = make_fmatrix4_scaling(outXVoxelSize, outYVoxelSize, outZVoxelSize);
  fmatrix4 revertDimension = make_fmatrix4_scaling(1./inXVoxelSize, 1./inYVoxelSize, 1./inZVoxelSize);

  // Calculate the final transformation. Since all transformations are applied
  // to the coordinate in reverse order, we have to do the multiplication
  // in reverse order too, to get the desired transformation
  fmatrix4 mat = moveBack * revertDimension * applyDimension * moveCenterToOrigin;

  out = transform(in, mat, _Size);
  thrust::copy(out.begin(), out.end(), output->begin());

  newState->setOutput(output);
}

}

}
