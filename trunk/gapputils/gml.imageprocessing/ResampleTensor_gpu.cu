/*
 * ResampleTensor_gpu.cu
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#include "ResampleTensor.h"

#include <tbblas/tensor.hpp>
#include <tbblas/imgproc/transform.hpp>

namespace gml {

namespace imageprocessing {

ResampleTensorChecker::ResampleTensorChecker () {
  ResampleTensor test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Input, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Size, test);
  CHECK_MEMORY_LAYOUT2(Output, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void ResampleTensor::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  using namespace tbblas;
  using namespace tbblas::imgproc;

  typedef host_tensor_t::value_t value_t;
  typedef tensor<value_t, 4, true> tensor_t;

  if (getInput()) {

    if (_Size[3] != _Input->size()[3]) {
      dlog(Severity::Warning) << "Number of output channels must be equal to the number of input channels. Aborting!";
      return;
    }

    boost::shared_ptr<host_tensor_t> output(new host_tensor_t());
    tensor_t in = *getInput(), out;

    float inXVoxelSize = 1;
    float inYVoxelSize = 1;
    float inZVoxelSize = 1;

    float outXVoxelSize = (float)in.size()[0] / _Size[0];
    float outYVoxelSize = (float)in.size()[1] / _Size[1];
    float outZVoxelSize = (float)in.size()[2] / _Size[2];

    // Find centers (-0.5, because we rotate around the center of a voxel)
    value_t infxc = value_t(in.size()[0]) / 2.0-0.5;
    value_t infyc = value_t(in.size()[1]) / 2.0-0.5;
    value_t infzc = value_t(in.size()[2]) / 2.0-0.5;

    value_t outfxc = value_t(_Size[0]) / 2.0-0.5;
    value_t outfyc = value_t(_Size[1]) / 2.0-0.5;
    value_t outfzc = value_t(_Size[2]) / 2.0-0.5;

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
    *output = out;

    newState->setOutput(output);
  }


  if (getInputs()) {

    v_host_tensor_t& inputs = *getInputs();
    boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

    for (size_t iTensor = 0; iTensor < inputs.size(); ++iTensor) {
      host_tensor_t& input = *inputs[iTensor];

      if (_Size[3] != input.size()[3]) {
        dlog(Severity::Warning) << "Number of output channels must be equal to the number of input channels. Aborting!";
        return;
      }

      boost::shared_ptr<host_tensor_t> output(new host_tensor_t());
      tensor_t in = input, out;

      float inXVoxelSize = 1;
      float inYVoxelSize = 1;
      float inZVoxelSize = 1;

      float outXVoxelSize = (float)in.size()[0] / _Size[0];
      float outYVoxelSize = (float)in.size()[1] / _Size[1];
      float outZVoxelSize = (float)in.size()[2] / _Size[2];

      // Find centers (-0.5, because we rotate around the center of a voxel)
      value_t infxc = value_t(in.size()[0]) / 2.0-0.5;
      value_t infyc = value_t(in.size()[1]) / 2.0-0.5;
      value_t infzc = value_t(in.size()[2]) / 2.0-0.5;

      value_t outfxc = value_t(_Size[0]) / 2.0-0.5;
      value_t outfyc = value_t(_Size[1]) / 2.0-0.5;
      value_t outfzc = value_t(_Size[2]) / 2.0-0.5;

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
      *output = out;

      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  }
}

}

}
