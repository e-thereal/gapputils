/*
 * FgrbmWriter.cpp
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#include "FgrbmWriter.h"

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

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FgrbmWriter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(FgrbmModel, Input("FGRBM"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(Filename, Output("File"), Filename("FGRBM Model (*.fgrbm)"), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

FgrbmWriter::FgrbmWriter() : data(0) {
  WfeUpdateTimestamp
  setLabel("FgrbmWriter");

  Changed.connect(capputils::EventHandler<FgrbmWriter>(this, &FgrbmWriter::changedHandler));
}

FgrbmWriter::~FgrbmWriter() {
  if (data)
    delete data;
}

void FgrbmWriter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FgrbmWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FgrbmWriter();

  if (!capputils::Verifier::Valid(*this) || !getFgrbmModel())
    return;

  std::cout << "Mean = " << getFgrbmModel()->getVisibleMean() << std::endl;
  capputils::Serializer::writeToFile(*getFgrbmModel(), getFilename());
}

void FgrbmWriter::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
