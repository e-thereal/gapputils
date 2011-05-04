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
  MainWindow w;
  w.show();
  ret = a.exec();
  Xmlizer::ToXml("gapphost.conf.xml", model);
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
