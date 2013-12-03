/*
 * InvWarp.cpp
 *
 *  Created on: Nov 27, 2013
 *      Author: tombr
 */

#include "InvWarp.h"

#include <capputils/Executer.h>

namespace gml {

namespace fsl {

BeginPropertyDefinitions(InvWarp)

  ReflectableBase(DefaultWorkflowElement<InvWarp>)

  WorkflowProperty(Reference, Input("Ref"), Filename(), FileExists())
  WorkflowProperty(Warpfield, Input("Warp"), Filename(), FileExists())
  WorkflowProperty(OutputName, Filename(), NotEmpty<Type>())
  WorkflowProperty(ProgramName, NotEmpty<Type>())
  WorkflowProperty(InverseField, Output(""), Filename())

EndPropertyDefinitions

InvWarp::InvWarp() : _ProgramName("invwarp") {
  setLabel("invwarp");
}

void InvWarp::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();
  capputils::Executer executer;
  executer.getCommand() << getProgramName() << " --ref=\"" << getReference() << "\" --warp=\"" << getWarpfield() << "\" --out=\"" << getOutputName() << "\"";
  dlog(Severity::Trace) << "Executing: " << executer.getCommandString();
  int returnValue = executer.execute();
  if (returnValue) {
    dlog(Severity::Error) << "Command failed with error code: " << returnValue;
    dlog(Severity::Error) << executer.getOutput();
  } else {
    dlog(Severity::Trace) << executer.getOutput();
  }

  newState->setInverseField(getOutputName());
}

}

} /* namespace gml */
