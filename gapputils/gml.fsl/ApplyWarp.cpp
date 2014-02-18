/*
 * ApplyWarp.cpp
 *
 *  Created on: Nov 27, 2013
 *      Author: tombr
 */

#include "ApplyWarp.h"

#include <capputils/Executer.h>

namespace gml {

namespace fsl {

BeginPropertyDefinitions(ApplyWarp)

  ReflectableBase(DefaultWorkflowElement<ApplyWarp>)

  WorkflowProperty(Reference, Input("Ref"), Filename(), FileExists())
  WorkflowProperty(Input, Input("In"), Filename(), FileExists())
  WorkflowProperty(Warpfield, Input("Warp"), Filename(), FileExists())
  WorkflowProperty(OutputName, Filename(), NotEmpty<Type>())
  WorkflowProperty(ProgramName, NotEmpty<Type>())
  WorkflowProperty(Output, Output("Out"), Filename())

EndPropertyDefinitions

ApplyWarp::ApplyWarp() : _ProgramName("applywarp") {
  setLabel("warp");
}

void ApplyWarp::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();
  capputils::Executer executer;
  executer.getCommand() << getProgramName() << " --ref=\"" << getReference() << "\" --in=\"" << getInput() << "\" --warp=\"" << getWarpfield() << "\" --out=\"" << getOutputName() << "\" --interp=sinc";
  dlog(Severity::Trace) << "Executing: " << executer.getCommandString();
  int returnValue = executer.execute();
  if (returnValue) {
    dlog(Severity::Error) << "Command failed with error code: " << returnValue;
    dlog(Severity::Error) << executer.getOutput();
  } else {
    dlog(Severity::Trace) << executer.getOutput();
  }

  newState->setOutput(getOutputName());
}

} /* namespace fsl */

} /* namespace gml */
