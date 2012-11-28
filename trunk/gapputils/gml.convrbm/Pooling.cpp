/*
 * Pooling.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "Pooling.h"

#include <algorithm>

namespace gml {
namespace convrbm {

BeginPropertyDefinitions(Pooling)
  ReflectableBase(DefaultWorkflowElement<Pooling>)

  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(BlockSize)
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(Outputs, Output("Ts"))
EndPropertyDefinitions

Pooling::Pooling() : _BlockSize(2) {
  setLabel("Pooling");
}

Pooling::~Pooling() { }

size_t count(const Model::tensor_t::dim_t& size) {
  size_t count = 1;
  for (unsigned i = 0; i < Model::dimCount; ++i)
    count *= size[i];
  return count;
}

void Pooling::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef tensor_t::dim_t dim_t;
  const unsigned dimCount = Model::dimCount;

  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getInputs();

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<tensor_t> >());

  dim_t inSize = inputs[0]->size(), inBlock, outBlock;

  if (getDirection() == CodingDirection::Encode) {
    inBlock = dim_t(getBlockSize());
    inBlock[dimCount - 1] = 1;

    outBlock = dim_t(1);
    outBlock[dimCount - 1] = count(inBlock);
  } else {
    outBlock = dim_t(getBlockSize());
    outBlock[dimCount - 1] = 1;

    inBlock = dim_t(1);
    inBlock[dimCount - 1] = count(outBlock);
  }

  dim_t outSize = inSize / inBlock * outBlock;

  for (size_t i = 0; i < inputs.size(); ++i) {
    boost::shared_ptr<tensor_t> output(new tensor_t(outSize));
    tensor_t& input = *inputs[i];

    for (int iz = 0, oz = 0; iz < inSize[2]; iz += inBlock[2], oz += outBlock[2]) {
      for (int iy = 0, oy = 0; iy < inSize[1]; iy += inBlock[1], oy += outBlock[1]) {
        for (int ix = 0, ox = 0; ix < inSize[0]; ix += inBlock[0], ox += outBlock[0]) {
          std::copy(input[seq(ix, iy, iz), inBlock].begin(), input[seq(ix, iy, iz), inBlock].end(),
              (*output)[seq(ox, oy, oz), outBlock].begin());
        }
      }
    }

    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

} /* namespace convrbm */
} /* namespace gml */
