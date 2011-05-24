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
  DefineProperty(Filename1, Observe(filename1Id = PROPERTY_ID), Input("File1"), Filename(), FileExists())
  DefineProperty(Filename2, Observe(filename2Id = PROPERTY_ID), Input("File2"), Filename())

EndPropertyDefinitions

int GenericViewer::filename1Id;
int GenericViewer::filename2Id;

void killProcess(QProcess* process) {
#ifdef Q_OS_LINUX2
    Q_PID pid=process->pid();
    QProcess killer;
    QStringList params;
    params << "--ppid";
    params << QString::number(pid);
    params << "-o";
    params << "pid";
    params << "--noheaders";
    killer.start("/bin/ps",params,QIODevice::ReadOnly);
    if(killer.waitForStarted(-1))
    {
        if(killer.waitForFinished(-1))
        {
            QByteArray temp=killer.readAllStandardOutput();
            QString str=QString::fromLocal8Bit(temp);
            QStringList list=str.split("\n");

            for(int i=0;i<list.size();i++)
            {
                if(!list.at(i).isEmpty())
                    ::kill(list.at(i).toInt(),SIGKILL);
            }
        }
    }
#endif
    process->terminate();
    process->waitForFinished(100);
    if (process->state() == QProcess::Running) {
      process->kill();
      process->waitForFinished(100);
    }
}

GenericViewer::GenericViewer() {
  setLabel("Viewer");
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

  if (eventId != filename1Id && eventId != filename2Id)
    return;

  if (viewer.state() == QProcess::Running) {
    killProcess(&viewer);
  }

  std::stringstream command;
  command << getProgram().c_str() << " \"" << getFilename1().c_str() << "\"";
  if (getFilename2().size())
    command << " \"" << getFilename2() << "\"";

  cout << "Executing: " << command.str().c_str() << endl;

  viewer.start(command.str().c_str());
  viewer.waitForStarted(100);
}

}
