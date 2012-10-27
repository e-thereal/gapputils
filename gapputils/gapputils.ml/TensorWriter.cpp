/*
 * TensorWriter.cpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#include "TensorWriter.h"

#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotNullAttribute.h>
#include <capputils/NotEmptyAttribute.h>

#include <capputils/Serializer.h>

#include "tbblas_serialize.hpp"

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(TensorWriter)
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<TensorWriter>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), Serialize<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

TensorWriter::TensorWriter() {

}

TensorWriter::~TensorWriter() {
}

void TensorWriter::update(gapputils::workflow::IProgressMonitor* monitor) const {
  capputils::Serializer::WriteToFile(*this, findProperty("Tensors"), getFilename());
  newState->setOutputName(getFilename());
}

} /* namespace ml */

} /* namespace gapputils */
