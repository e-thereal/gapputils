/*
 * True3dViewer.cpp
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#include "true3dviewer.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace true3d {

BeginPropertyDefinitions(True3dViewer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

EndPropertyDefinitions

True3dViewer::True3dViewer() : data(0)
{
  WfeUpdateTimestamp
  setLabel("True3dViewer");

  Changed.connect(capputils::EventHandler<True3dViewer>(this, &True3dViewer::changedHandler));
}

True3dViewer::~True3dViewer() {
  if (data)
    delete data;
}

void True3dViewer::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void True3dViewer::execute(gapputils::workflow::IProgressMonitor* monitor) const {

  if (!data)
    data = new True3dViewer();

  if (!capputils::Verifier::Valid(*this))
    return;

  
}

void True3dViewer::writeResults() {
  if (!data)
    return;
}

void True3dViewer::show() {
  std::cout << "Showing OpenGL window with Qt." << std::endl;
}

}
