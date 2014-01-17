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

namespace gml {

namespace rbm {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename("RBM Model (*.rbm)"), FileExists())
  WorkflowProperty(Model, Output("RBM"))
  WorkflowProperty(VisibleCount, NoParameter())
  WorkflowProperty(HiddenCount, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader() : _VisibleCount(0), _HiddenCount(0) {
  setLabel("Reader");
}

void ModelReader::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();
  using namespace tbblas;

  typedef Model::matrix_t matrix_t;
  typedef Model::value_t value_t;

  boost::shared_ptr<Model> rbm(new Model());
  Serializer::readFromFile(*rbm, getFilename());

  // Compatibility with unmasked model
  if (!rbm->getVisibleMask())
    rbm->setVisibleMask(boost::make_shared<matrix_t>(ones<value_t>(1, rbm->getWeightMatrix()->size()[0])));

  newState->setModel(rbm);
  newState->setVisibleCount(rbm->getWeightMatrix()->size()[0]);
  newState->setHiddenCount(rbm->getWeightMatrix()->size()[1]);
  newState->setVisibleUnitType(rbm->getVisibleUnitType());
  newState->setHiddenUnitType(rbm->getHiddenUnitType());
}

}

}
