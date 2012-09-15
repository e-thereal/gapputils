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
  WorkflowProperty(Filename, Filename(), FileExists())
  WorkflowProperty(Tensors, Output("Ts"), Serialize<Type>())

EndPropertyDefinitions

TensorReader::TensorReader() {
}

TensorReader::~TensorReader() {
}

void TensorReader::update(gapputils::workflow::IProgressMonitor* monitor) const {
  capputils::Serializer::ReadFromFile(*newState, newState->findProperty("Tensors"), getFilename());
}

} /* namespace ml */

} /* namespace gapputils */
