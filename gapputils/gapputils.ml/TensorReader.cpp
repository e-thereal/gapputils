/*
 * TensorReader.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "TensorReader.h"

#include <capputils/FileExistsAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/NotEmptyAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Serializer.h>

#include "tbblas_serialize.hpp"

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(TensorReader)
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<TensorReader>)
  WorkflowProperty(Filename, Input("File"), Filename(), FileExists())
  WorkflowProperty(Tensors, Output("Ts"), Serialize<Type>())
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(Count, NoParameter())

EndPropertyDefinitions

TensorReader::TensorReader() : _Width(0), _Height(0), _Depth(0), _Count(0) {
  setLabel("Reader");
}

TensorReader::~TensorReader() {
}

void TensorReader::update(gapputils::workflow::IProgressMonitor* monitor) const {
  capputils::Serializer::ReadFromFile(*newState, newState->findProperty("Tensors"), getFilename());
  std::vector<boost::shared_ptr<tensor_t> >& tensors = *newState->getTensors();
  if (tensors.size()) {
    newState->setWidth(tensors[0]->size()[0]);
    newState->setHeight(tensors[0]->size()[1]);
    newState->setDepth(tensors[0]->size()[2]);
    newState->setCount(tensors.size());
  }
}

} /* namespace ml */

} /* namespace gapputils */
