/*
 * ConvRbmWriter.cpp
 *
 *  Created on: Apr 09, 2012
 *      Author: tombr
 */

#include "ConvRbmWriter.h"

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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int ConvRbmWriter::inputId;

BeginPropertyDefinitions(ConvRbmWriter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Model, Input("CRBM"), Volatile(), ReadOnly(), Observe(inputId = Id), TimeStamp(Id))
  DefineProperty(Filename, Output("File"), Filename("CRBM Model (*.crbm)"), Observe(Id), TimeStamp(Id))
  DefineProperty(AutoSave, Observe(Id))

EndPropertyDefinitions

ConvRbmWriter::ConvRbmWriter() : _AutoSave(false), data(0) {
  WfeUpdateTimestamp
  setLabel("Writer");

  Changed.connect(capputils::EventHandler<ConvRbmWriter>(this, &ConvRbmWriter::changedHandler));
}

ConvRbmWriter::~ConvRbmWriter() {
  if (data)
    delete data;
}

void ConvRbmWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAutoSave()) {
    execute(0);
    writeResults();
  }
}

void ConvRbmWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ConvRbmWriter();

  assert(getHostInterface());

  if (!capputils::Verifier::Valid(*this) || !getModel())
    return;

  capputils::Serializer::writeToFile(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
}

void ConvRbmWriter::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
