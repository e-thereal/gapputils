/*
 * MakeTensors.cpp
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#include "MakeTensors.h"

#include <algorithm>

#include <capputils/attributes/DeprecatedAttribute.h>
#include <capputils/attributes/RenamedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(MakeTensors, Renamed("gml::imaging::core::MakeTensors"), Deprecated("Use gml::imaging::core::MakeTensors instead."))

  ReflectableBase(DefaultWorkflowElement<MakeTensors>)

  WorkflowProperty(Vectors, Input("Vs"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Width)
  WorkflowProperty(Height)
  WorkflowProperty(Depth)
  WorkflowProperty(ChannelCount)
  WorkflowProperty(Tensors, Output("Ts"))
  WorkflowProperty(Tensor, Output("T"))

EndPropertyDefinitions

MakeTensors::MakeTensors() : _Width(0), _Height(0), _Depth(0), _ChannelCount(0) {
  setLabel("D2T");
}

void MakeTensors::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::vector<boost::shared_ptr<std::vector<double> > >& inputs = *getVectors();
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<tensor_t> >);

  for (size_t i = 0; i < inputs.size(); ++i) {
    boost::shared_ptr<tensor_t> output(new tensor_t(getWidth(), getHeight(), getDepth(), getChannelCount()));
    if (output->count() != inputs[i]->size()) {
      dlog(Severity::Warning) << "Element count of the tensor doesn't match the data vector size. Skipping data vector.";
      continue;
    }
    std::copy(inputs[i]->begin(), inputs[i]->end(), output->begin());
    outputs->push_back(output);
  }

  newState->setTensors(outputs);
  newState->setTensor(outputs->at(0));
}

} /* namespace convrbm4d */

} /* namespace gml */
