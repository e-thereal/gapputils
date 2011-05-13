/*
 * GenericViewer.cpp
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#include "GenericViewer.h"

#include <FilenameAttribute.h>
#include "InputAttribute.h"
#include <FileExists.h>
#include <ObserveAttribute.h>
#include <Verifier.h>
#include <sstream>
#include <cstdlib>
#include <EventHandler.h>
#include <VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(GenericViewer)

  ReflectableBase(DefaultWorkflowElement)
  DefineProperty(Program, Observe(PROPERTY_ID))
  DefineProperty(Filename, Observe(filenameId = PROPERTY_ID), Input(), Filename(), FileExists(), Volatile())

EndPropertyDefinitions

int GenericViewer::filenameId;

GenericViewer::GenericViewer() {
  setLabel("Viewer");
  Changed.connect(capputils::EventHandler<GenericViewer>(this, &GenericViewer::changeHandler));
}

GenericViewer::~GenericViewer() {
}

void GenericViewer::changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == filenameId && capputils::Verifier::Valid(*this)) {
    std::stringstream command;
    command << getProgram().c_str() << " \"" << getFilename().c_str() << "\" &";
    std::system(command.str().c_str());
  }
}

}
