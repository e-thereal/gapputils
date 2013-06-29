/*
 * Pooling.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "Pooling.h"

#include <algorithm>

#include <tbblas/rearrange.hpp>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Pooling)
  ReflectableBase(DefaultWorkflowElement<Pooling>)

  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(BlockWidth)
  WorkflowProperty(BlockHeight)
  WorkflowProperty(BlockDepth)
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(Outputs, Output("Ts"))
EndPropertyDefinitions

Pooling::Pooling() : _BlockWidth(2), _BlockHeight(2), _BlockDepth(2) {
  setLabel("Pooling");
}

Pooling::~Pooling() { }

int count(const Model::tensor_t::dim_t& size) {
  int count = 1;
  for (unsigned i = 0; i < Model::dimCount; ++i)
    count *= size[i];
  return count;
}

void Pooling::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor_t::dim_t dim_t;

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > inputs = getInputs();
  const bool cleanup = getAtomicWorkflow() && inputs.use_count() == 2;

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<tensor_t> >());

  dim_t inSize = inputs->at(0)->size(), inBlock, outBlock;

  if (getDirection() == CodingDirection::Encode) {
    inBlock = seq(getBlockWidth(), getBlockHeight(), getBlockDepth(), 1);
    outBlock = seq(1, 1, 1, count(inBlock));
  } else {
    outBlock = seq(getBlockWidth(), getBlockHeight(), getBlockDepth(), 1);
    inBlock = seq(1, 1, 1, count(outBlock));
  }

  dim_t outSize = inSize / inBlock * outBlock;

  for (size_t i = 0; i < inputs->size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    boost::shared_ptr<tensor_t> output(new tensor_t(outSize));
    boost::shared_ptr<tensor_t> input = inputs->at(i);
    if (cleanup)
      inputs->at(i) = boost::shared_ptr<tensor_t>();

    if (getMethod() == PoolingMethod::StackPooling) {
      for (int ik = 0, ok = 0; ik < inSize[3]; ik += inBlock[3], ok += outBlock[3]) {
        for (int iz = 0, oz = 0; iz < inSize[2]; iz += inBlock[2], oz += outBlock[2]) {
          for (int iy = 0, oy = 0; iy < inSize[1]; iy += inBlock[1], oy += outBlock[1]) {
            for (int ix = 0, ox = 0; ix < inSize[0]; ix += inBlock[0], ox += outBlock[0]) {
              std::copy((*input)[seq(ix, iy, iz, ik), inBlock].begin(), (*input)[seq(ix, iy, iz, ik), inBlock].end(),
                  (*output)[seq(ox, oy, oz, ok), outBlock].begin());
            }
          }
        }
      }
    } else {
      if (getDirection() == CodingDirection::Encode)
        *output = rearrange(*input, inBlock);
      else
        *output = rearrange_r(*input, outBlock);
    }

    outputs->push_back(output);
    if (monitor)
      monitor->reportProgress(100. * i / inputs->size());
  }

  newState->setOutputs(outputs);
}

} /* namespace convrbm4d */
} /* namespace gml */
