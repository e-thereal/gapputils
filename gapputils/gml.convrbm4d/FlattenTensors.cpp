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

  WorkflowProperty(Tensor, Input("T"))
  WorkflowProperty(Tensors, Input("Ts"))
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
  Logbook& dlog = getLogbook();

  if (!getTensor() && !(getTensors() && getTensors()->size())) {
    dlog(Severity::Warning) << "No tensors given. Aborting!";
    return;
  }

  if (getMode() == FlattenMode::OneVectorPerTensor) {
    boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > outputs(
        new std::vector<boost::shared_ptr<std::vector<double> > >());

    if (getTensor()) {
      tensor_t& input = *getTensor();
      boost::shared_ptr<std::vector<double> > output(new std::vector<double>(input.count()));
      std::copy(input.begin(), input.end(), output->begin());
      outputs->push_back(output);
    }

    if (getTensors() && getTensors()->size()) {
      std::vector<boost::shared_ptr<tensor_t> >& inputs = *getTensors();
      for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
        boost::shared_ptr<std::vector<double> > output(new std::vector<double>(inputs[i]->count()));
        std::copy(inputs[i]->begin(), inputs[i]->end(), output->begin());
        outputs->push_back(output);
        if (monitor)
          monitor->reportProgress(100.0 * i / inputs.size());
      }
    }
    newState->setVectors(outputs);
  } else {

    const size_t count = (getTensor() ? getTensor()->count() : getTensors()->at(0)->count());
    size_t tensorCount = 0;
    if (getTensor())
      ++tensorCount;
    if (getTensors())
      tensorCount += getTensors()->size();

    boost::shared_ptr<std::vector<double> > output(new std::vector<double>(tensorCount * count));

    int iOut = 0;
    if (getTensor()) {
      tensor_t& input = *getTensor();
      std::copy(input.begin(), input.end(), output->begin() + iOut++ * count);
    }

    if (getTensors() && getTensors()->size()) {
      std::vector<boost::shared_ptr<tensor_t> >& inputs = *getTensors();
      for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
        std::copy(inputs[i]->begin(), inputs[i]->end(), output->begin() + iOut++ * count);
        if (monitor)
          monitor->reportProgress(100.0 * i / inputs.size());
      }
    }
    newState->setVector(output);
  }

  tensor_t& input = (getTensor() ? *getTensor() : *getTensors()->at(0));

  newState->setWidth(input.size()[0]);
  newState->setHeight(input.size()[1]);
  newState->setDepth(input.size()[2]);
  newState->setChannelCount(input.size()[3]);
  newState->setCount(input.count());
}

} /* namespace convrbm4d */
} /* namespace gml */
