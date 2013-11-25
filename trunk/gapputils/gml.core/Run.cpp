/*
 * Run.cpp
 *
 *  Created on: Nov 11, 2013
 *      Author: tombr
 */

#include "Run.h"

#include <capputils/Executer.h>

#include <boost/regex.hpp>

namespace gml {

namespace core {

BeginPropertyDefinitions(Run, Description("Runs arbitrary commands on the command line."))

  ReflectableBase(DefaultWorkflowElement<Run>)

  WorkflowProperty(Input, Input("In"), Description("Hold the name of the primary input"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Description("Name of the output file. The place holder $input will be replaced by the content of the Input property."), Filename(), NotEmpty<Type>())
  WorkflowProperty(Command, Description("Command string. The place holders $input and $output will be replaced by the content of the Input and Output property."))
  WorkflowProperty(Output, Output("Out"), Description("Will be set according to the pattern specified in OutputName."))

EndPropertyDefinitions

Run::Run() {
  setLabel("Run");
}

void Run::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::string outputName = boost::regex_replace(getOutputName(), boost::regex("\\$input"), getInput());
  std::string command = boost::regex_replace(getCommand(), boost::regex("\\$input"), getInput());
  command = boost::regex_replace(command, boost::regex("\\$output"), outputName);
  dlog(Severity::Trace) << "Running command: " << command;

  capputils::Executer executer;
  executer.getCommand() << command;
  int returnValue = executer.execute();
  if (returnValue) {
    dlog(Severity::Error) << "Command failed with error code: " << returnValue;
    dlog(Severity::Error) << executer.getOutput();
  } else {
    dlog(Severity::Trace) << executer.getOutput();
  }

  newState->setOutput(outputName);
}

} /* namespace core */

} /* namespace gml */
