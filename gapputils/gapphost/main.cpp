#include "MainWindow.h"
#include <QtGui/QApplication>

//#include <cublas.h>
#include <Xmlizer.h>
#include <ArgumentsParser.h>
#include <Verifier.h>
#include <iostream>
#include <gapputils.h>

//#include "../GaussianProcesses/Paper.h"
#include "DataModel.h"

using namespace gapputils::host;
using namespace gapputils;
using namespace capputils;
using namespace std;

//#define AUTOTEST

int main(int argc, char *argv[])
{
  //cublasInit();
  registerClasses();

  int ret = 0;
#ifndef AUTOTEST
  QApplication a(argc, argv);
  DataModel& model = DataModel::getInstance();
  Xmlizer::FromXml(model, "gapphost.conf.xml");
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
  if (!model.getNoGui()) {
    w.show();
    ret = a.exec();
  }

  TiXmlElement* modelElement = Xmlizer::CreateXml(model);
  TiXmlElement* mainWorkflowElement = new TiXmlElement("MainWorkflow");
  TiXmlElement* workflowElement = model.getMainWorkflow()->getXml(false);
  Xmlizer::ToXml(*workflowElement, *model.getMainWorkflow());

  mainWorkflowElement->LinkEndChild(workflowElement);
  modelElement->LinkEndChild(mainWorkflowElement);

  Xmlizer::ToFile("gapphost.conf.xml", modelElement);

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
