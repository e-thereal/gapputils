#include "MainWindow.h"
#include <QtGui/QApplication>

//#include <cublas.h>
#include <capputils/Xmlizer.h>
#include <capputils/ArgumentsParser.h>
#include <capputils/Verifier.h>
#include <iostream>
#include <capputils/ReflectableClassFactory.h>

//#include "../GaussianProcesses/Paper.h"
#include "DataModel.h"
#include "Workflow.h"
#include "DefaultInterface.h"

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;

//#define AUTOTEST

int main(int argc, char *argv[])
{
  //cublasInit();
  int ret = 0;
#ifndef AUTOTEST
  QApplication a(argc, argv);
  DataModel& model = DataModel::getInstance();
  Xmlizer::FromXml(model, "gapphost.conf.xml");

  // Initialize if necessary
  if (!model.getMainWorkflow())
    model.setMainWorkflow(new Workflow());
  if (!model.getMainWorkflow()->getModule())
    model.getMainWorkflow()->setModule(new DefaultInterface());

  reflection::ReflectableClass& wfModule = *model.getMainWorkflow()->getModule();

  ArgumentsParser::Parse(model, argc, argv);
  ArgumentsParser::Parse(wfModule, argc, argv);
  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage("gapphost", model);
    ArgumentsParser::PrintUsage("Workflow switches:", wfModule);
    return 0;
  }

  model.getMainWorkflow()->resumeFromModel();

  MainWindow w;
  if (model.getNoGui()) {
    // TODO: wait until update has finished.
    model.getMainWorkflow()->updateOutputs();
  } else {
    w.show();
    ret = a.exec();
  }

  model.saveToFile("gapphost.conf.xml");
  delete model.getMainWorkflow();

#else
  Paper paper;
  paper.setRun(0);
  ArgumentsParser::Parse(paper, argc, argv);
  Xmlizer::FromXml(paper, paper.getConfigurationName());
  paper.setRun(1);
  Xmlizer::ToXml(paper.getConfigurationName(), paper);
#endif
  //cublasShutdown();
  return ret;
}
