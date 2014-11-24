/*
 * RbmReader.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "ModelReader.h"

#include <capputils/Serializer.h>
#include <tbblas/ones.hpp>
#include <tbblas/io.hpp>

#include <tbblas/deeplearn/serialize.hpp>

namespace gml {

namespace rbm {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename("RBM Model (*.rbm)"), FileExists())
  WorkflowProperty(Model, Output("RBM"))
  WorkflowProperty(FloatModel, Flag())
  WorkflowProperty(VisibleCount, NoParameter())
  WorkflowProperty(HiddenCount, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader() : _FloatModel(false), _VisibleCount(0), _HiddenCount(0) {
  setLabel("Reader");
}

void ModelReader::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<model_t> model;

  if (getFloatModel()) {
    tbblas::deeplearn::rbm_model<float> fmodel;
    tbblas::deeplearn::deserialize(getFilename(), fmodel);
    model = boost::make_shared<model_t>(fmodel);
  } else {
    model = boost::make_shared<model_t>();
    tbblas::deeplearn::deserialize(getFilename(), *model);
  }

  newState->setModel(model);
  newState->setVisibleCount(model->weights().size()[0]);
  newState->setHiddenCount(model->weights().size()[1]);
  newState->setVisibleUnitType(model->visibles_type());
  newState->setHiddenUnitType(model->hiddens_type());
}

}

}
