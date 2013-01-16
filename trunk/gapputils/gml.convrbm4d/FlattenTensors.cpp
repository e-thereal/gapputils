/*
 * FlattenTensors.cpp
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#include "FlattenTensors.h"

#include <algorithm>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(FlattenTensors)

  ReflectableBase(DefaultWorkflowElement<FlattenTensors>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mode, Enumerator<Type>())
  WorkflowProperty(Vectors, Output("Vs"))
  WorkflowProperty(Vector, Output("D"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(ChannelCount, NoParameter())
  WorkflowProperty(Count, NoParameter())

EndPropertyDefinitions

FlattenTensors::FlattenTensors() : _Width(0), _Height(0), _Depth(0), _ChannelCount(0), _Count(0) {
  setLabel("T2D");
}

void FlattenTensors::update(IProgressMonitor* monitor) const {
  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getTensors();

  if (getMode() == FlattenMode::OneVectorPerTensor) {
    boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > outputs(
        new std::vector<boost::shared_ptr<std::vector<double> > >());

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
      boost::shared_ptr<std::vector<double> > output(new std::vector<double>(inputs[i]->count()));
      std::copy(inputs[i]->begin(), inputs[i]->end(), output->begin());
      outputs->push_back(output);
      if (monitor)
        monitor->reportProgress(100.0 * i / inputs.size());
    }
    newState->setVectors(outputs);
  } else {
    const size_t count = inputs[0]->count();
    boost::shared_ptr<std::vector<double> > output(new std::vector<double>(inputs.size() * count));
    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
      std::copy(inputs[i]->begin(), inputs[i]->end(), output->begin() + i * count);
      if (monitor)
        monitor->reportProgress(100.0 * i / inputs.size());
    }
    newState->setVector(output);
  }

  newState->setWidth(inputs[0]->size()[0]);
  newState->setHeight(inputs[0]->size()[1]);
  newState->setDepth(inputs[0]->size()[2]);
  newState->setChannelCount(inputs[0]->size()[3]);
  newState->setCount(inputs[0]->count());
}

} /* namespace convrbm4d */
} /* namespace gml */
