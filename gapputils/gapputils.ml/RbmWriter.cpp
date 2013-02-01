/*
 * RbmWriter.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "RbmWriter.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Serializer.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int RbmWriter::inputId;

BeginPropertyDefinitions(RbmWriter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(RbmModel, Input("RBM"), Volatile(), ReadOnly(), Observe(inputId = Id), TimeStamp(Id))
  DefineProperty(Filename, Output("File"), Filename("RBM Model (*.rbm)"), Observe(Id), TimeStamp(Id))
  DefineProperty(AutoSave, Observe(Id))

EndPropertyDefinitions

RbmWriter::RbmWriter() : _AutoSave(false), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmWriter");

  Changed.connect(capputils::EventHandler<RbmWriter>(this, &RbmWriter::changedHandler));
}

RbmWriter::~RbmWriter() {
  if (data)
    delete data;
}

void RbmWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAutoSave()) {
    execute(0);
    writeResults();
  }
}

void RbmWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmWriter();

  if (!capputils::Verifier::Valid(*this) || !getRbmModel())
    return;

  capputils::Serializer::writeToFile(*getRbmModel(), getFilename());
}

void RbmWriter::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
