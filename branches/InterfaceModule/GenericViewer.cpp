/*
 * GenericViewer.cpp
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#include "GenericViewer.h"

#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/FileExists.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/Verifier.h>
#include <sstream>
#include <cstdlib>
#include <capputils/EventHandler.h>
#include <capputils/VolatileAttribute.h>

#include <iostream>
#include <signal.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

BeginPropertyDefinitions(GenericViewer)

  ReflectableBase(workflow::WorkflowElement)
  DefineProperty(Program, Observe(PROPERTY_ID))
  DefineProperty(Filename1, Observe(filename1Id = PROPERTY_ID), Input("File1"), Filename(), FileExists(), Volatile())
  DefineProperty(Filename2, Observe(filename2Id = PROPERTY_ID), Input("File2"), Filename(), Volatile())
  DefineProperty(Filename3, Observe(filename3Id = PROPERTY_ID), Input("File3"), Filename(), Volatile())

EndPropertyDefinitions

int GenericViewer::filename1Id;
int GenericViewer::filename2Id;
int GenericViewer::filename3Id;

void killProcess(QProcess* process) {
  process->terminate();
  process->waitForFinished(100);
  if (process->state() == QProcess::Running) {
    process->kill();
    process->waitForFinished(100);
  }
}

GenericViewer::GenericViewer() {
  setLabel("Viewer");
  updateViewTimer.setSingleShot(true);
  updateViewTimer.setInterval(1000);
  connect(&updateViewTimer, SIGNAL(timeout()), this, SLOT(updateView()));
  Changed.connect(capputils::EventHandler<GenericViewer>(this, &GenericViewer::changedHandler));
}

GenericViewer::~GenericViewer() {
  if (viewer.state() == QProcess::Running) {
    killProcess(&viewer);
  }
}

void GenericViewer::changedHandler(capputils::ObservableClass*, int eventId) {
  if (!capputils::Verifier::Valid(*this))
    return;

  if (eventId != filename1Id && eventId != filename2Id && eventId != filename3Id)
    return;

  updateViewTimer.start();
}

void GenericViewer::writeResults() {
  updateView();
}

void GenericViewer::updateView() {
  if (!capputils::Verifier::Valid(*this))
    return;

  if (getFilename2().size() && !FileExistsAttribute::exists(getFilename2()))
    return;

  if (getFilename3().size() && !FileExistsAttribute::exists(getFilename3()))
    return;

  if (viewer.state() == QProcess::Running) {
    killProcess(&viewer);
  }

  std::stringstream command;
  command << getProgram().c_str() << " \"" << getFilename1().c_str() << "\"";
  if (getFilename2().size())
    command << " \"" << getFilename2() << "\"";
  if (getFilename3().size())
    command << " \"" << getFilename3() << "\"";

  cout << "Executing: " << command.str().c_str() << endl;

  viewer.start(command.str().c_str());
  viewer.waitForStarted(100);
}

}
