/*
 * FilterPatch_gpu.cu
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "FilterPatch.h"

#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>
#include <tbblas/rearrange.hpp>

#include <omp.h>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/conv_rbm.hpp>
#include <tbblas/deeplearn/conv_rbm_model.hpp>

namespace gml {

namespace convrbm4d {

FilterPatchChecker::FilterPatchChecker() {
  FilterPatch filterPatch;
  filterPatch.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, filterPatch);
  CHECK_MEMORY_LAYOUT2(Inputs, filterPatch);
  CHECK_MEMORY_LAYOUT2(Direction, filterPatch);
  CHECK_MEMORY_LAYOUT2(SuperPatchWidth, filterPatch);
  CHECK_MEMORY_LAYOUT2(SuperPatchHeight, filterPatch);
  CHECK_MEMORY_LAYOUT2(SuperPatchDepth, filterPatch);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, filterPatch);
  CHECK_MEMORY_LAYOUT2(GpuCount, filterPatch);
  CHECK_MEMORY_LAYOUT2(DoubleWeights, filterPatch);
  CHECK_MEMORY_LAYOUT2(OnlyFilters, filterPatch);
  CHECK_MEMORY_LAYOUT2(SampleUnits, filterPatch);

  CHECK_MEMORY_LAYOUT2(Outputs, filterPatch);
}

unsigned int upper_power_of_two(unsigned int v);

//#define TRACE std::cout << __LINE__ << std::endl;

void FilterPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef float value_t;
  const unsigned dimCount = model_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  model_t& model = *getModel();

  if (getFilterBatchSize() > model.filters().size() ||
      model.filters().size() % getFilterBatchSize() != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  dim_t maxInputSize;

  if (getDirection() == CodingDirection::Encode) {
    maxInputSize = inputs[0]->size();
  } else {
    maxInputSize = seq<dimCount>(0);
    for (int z = 0, i = 0; z < model.pooling_size()[2]; ++z) {
      for (int y = 0; y < model.pooling_size()[1]; ++y) {
        for (int x = 0; x < model.pooling_size()[0]; ++x, ++i) {
          if (i >= inputs.size()) {
            dlog(Severity::Warning) << "Number of input images doesn't make sense. Aborting!";
            return;
          }

          maxInputSize = maxInputSize + inputs[i]->size() * seq(!y && !z, !x && !z, !x && !y, !x && !y && !z);
        }
      }
    }
    if (model.convolution_type() == convolution_type::Valid) {
      maxInputSize = maxInputSize + model.kernel_size() - 1;
      maxInputSize[dimCount - 1] = model.kernel_size()[dimCount - 1];
    }
  }

  // Change size of bias terms to the super patch size and unstride the model
  dim_t superPatchSize = seq(
      getSuperPatchWidth() > 0 ? getSuperPatchWidth() : maxInputSize[0],
      getSuperPatchHeight() > 0 ? getSuperPatchHeight() : maxInputSize[1],
      getSuperPatchDepth() > 0 ? getSuperPatchDepth() : maxInputSize[2],
      maxInputSize[3]);

  dim_t superPatchLayerSize = superPatchSize;
  superPatchLayerSize[dimCount - 1] = 1;

  dim_t patchSize = model.input_size();
  dim_t oldStride = model.stride_size();

  model.change_stride(seq<dimCount>(1));
  model.change_size(superPatchSize);

  dim_t superPatchHiddenSize = model.hiddens_size();
  dim_t superPatchHiddenLayerSize = superPatchHiddenSize;
  superPatchHiddenLayerSize[dimCount - 1] = 1;

  conv_rbm<float, 4> crbm(model, getGpuCount());
  crbm.set_batch_length(getFilterBatchSize());

  crbm.allocate_gpu_memory();

  tensor_t input, output;
  host_tensor_t overlapMask;

  std::vector<tensor_t> slices(model.pooling_size().prod());

  if (getDirection() == CodingDirection::Encode) {

    for (size_t iSample = 0; iSample < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
      input = *inputs[iSample];   // copies memory to the device. Makes rearranging faster

      dim_t inputSize, hiddenSize;
      hiddenSize = inputSize = inputs[iSample]->size();
      if (model.convolution_type() == convolution_type::Valid) {
        hiddenSize = hiddenSize - model.kernel_size() + 1;
        hiddenSize[dimCount - 1] = model.filters().size();
      }

      if (input.size()[3] != model.input_size()[3]) {
        dlog(Severity::Warning) << "Number of channels doesn't match. Aborting!";
        return;
      }

      output.resize(hiddenSize);

      for (int z = 0; z < hiddenSize[2]; z += superPatchHiddenSize[2]) {
        for (int y = 0; y < hiddenSize[1]; y += superPatchHiddenSize[1]) {
          for (int x = 0; x < hiddenSize[0]; x += superPatchHiddenSize[0]) {

            // Get new patch
            dim_t topleft = seq(x, y, z, 0);
            dim_t overlap = min(superPatchSize, inputSize - topleft);   // the overlap of the current super patch and the image
            dim_t hiddenOverlap = min(superPatchHiddenSize, hiddenSize - topleft);
            dim_t overlapMaskSize = overlap;
            overlapMaskSize[dimCount - 1] = 1;

            overlapMask = zeros<value_t>(superPatchLayerSize);
            overlapMask[seq<dimCount>(0), overlapMaskSize] = ones<value_t>(overlapMaskSize);
            crbm.change_mask(overlapMask);

            crbm.visibles() = zeros<value_t>(superPatchSize);
            crbm.visibles()[seq<dimCount>(0), overlap] = input[topleft, overlap];

            crbm.normalize_visibles();
            if (getSampleUnits())
              crbm.sample_hiddens();
            else
              crbm.infer_hiddens();

            output[topleft, hiddenOverlap] = crbm.hiddens()[seq<dimCount>(0), hiddenOverlap];
          }
        }
      }

      // Apply pooling
      for (int z = 0, i = 0; z < model.pooling_size()[2]; ++z) {
        for (int y = 0; y < model.pooling_size()[1]; ++y) {
          for (int x = 0; x < model.pooling_size()[0]; ++x, ++i) {
            slices[i] = output[seq(x,y,z,0), model.pooling_size(), output.size() - seq(x,y,z,0)];
            outputs->push_back(boost::make_shared<host_tensor_t>(slices[i]));
          }
        }
      }

      if (monitor)
        monitor->reportProgress(100. * iSample / inputs.size());
    }
  } else {
    for (size_t iSample = 0; iSample < inputs.size();) {

      dim_t inputSize, hiddenSize;
      inputSize = seq<dimCount>(0);
      for (int z = 0, i = 0; z < model.pooling_size()[2]; ++z) {
        for (int y = 0; y < model.pooling_size()[1]; ++y) {
          for (int x = 0; x < model.pooling_size()[0]; ++x, ++i) {
            if (i + iSample >= inputs.size()) {
              dlog(Severity::Warning) << "Number of input images doesn't make sense. Aborting!";
              return;
            }

            inputSize = inputSize + inputs[i + iSample]->size() * seq(!y && !z, !x && !z, !x && !y, !x && !y && !z);
          }
        }
      }
      hiddenSize = inputSize;
      if (model.convolution_type() == convolution_type::Valid) {
        inputSize = inputSize + model.kernel_size() - 1;
        inputSize[dimCount - 1] = model.kernel_size()[dimCount - 1];
      }

      // Revert pooling
      input.resize(hiddenSize);
      for (int z = 0; z < model.pooling_size()[2]; ++z) {
        for (int y = 0; y < model.pooling_size()[1]; ++y) {
          for (int x = 0; x < model.pooling_size()[0]; ++x, ++iSample) {
            if (iSample >= inputs.size()) {
              dlog(Severity::Warning) << "Number of input images doesn't make sense. Aborting!";
              return;
            }
            slices[iSample % slices.size()] = *inputs[iSample];
            input[seq(x,y,z,0), model.pooling_size(), input.size() - seq(x,y,z,0)] = slices[iSample % slices.size()];
          }
        }
      }

      output.resize(inputSize);

      for (int z = 0; z < hiddenSize[2]; z += superPatchHiddenSize[2]) {
        for (int y = 0; y < hiddenSize[1]; y += superPatchHiddenSize[1]) {
          for (int x = 0; x < hiddenSize[0]; x += superPatchHiddenSize[0]) {

            // Get new patch
            dim_t topleft = seq(x, y, z, 0);
            dim_t overlap = min(superPatchSize, inputSize - topleft);   // the overlap of the current super patch and the image
            dim_t hiddenOverlap = min(superPatchHiddenSize, hiddenSize - topleft);
            dim_t overlapMaskSize = overlap;
            overlapMaskSize[dimCount - 1] = 1;

            overlapMask = zeros<value_t>(superPatchLayerSize);
            overlapMask[seq<dimCount>(0), overlapMaskSize] = ones<value_t>(overlapMaskSize);
            crbm.change_mask(overlapMask);

//            crbm.visibles() = zeros<value_t>(superPatchSize);
//            crbm.visibles()[seq<dimCount>(0), overlap] = input[topleft, overlap];
            crbm.hiddens() = zeros<value_t>(superPatchHiddenSize);
            crbm.hiddens()[seq<dimCount>(0), hiddenOverlap] = input[topleft, hiddenOverlap];


//            crbm.normalize_visibles();
//            if (getSampleUnits())
//              crbm.sample_hiddens();
//            else
//              crbm.infer_hiddens();
            if (getSampleUnits())
              crbm.sample_visibles();
            else
              crbm.infer_visibles();
            crbm.diversify_visibles();

//            output[topleft, hiddenOverlap] = crbm.hiddens()[seq<dimCount>(0), hiddenOverlap];
            output[topleft, overlap] = crbm.visibles()[seq<dimCount>(0), overlap];
          }
        }
      }

      outputs->push_back(boost::make_shared<host_tensor_t>(output));
      if (monitor)
        monitor->reportProgress(100. * iSample / inputs.size());
    }
  }

  model.change_size(patchSize);
  model.change_stride(oldStride);

  newState->setOutputs(outputs);
}

}

}
