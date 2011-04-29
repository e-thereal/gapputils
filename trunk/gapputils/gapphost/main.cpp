#include "MainWindow.h"
#include <QtGui/QApplication>

#include <cublas.h>
#include <Xmlizer.h>
#include <ArgumentsParser.h>
#include <Verifier.h>

#include "Paper.h"
#include "DataModel.h"

using namespace gapputils::host;
using namespace gapputils;
using namespace capputils;

//#define AUTOTEST

int main(int argc, char *argv[])
{
  cublasInit();
  int ret = 0;
#ifndef AUTOTEST
  QApplication a(argc, argv);
  MainWindow w;
  w.show();
  ret = a.exec();
  DataModel& model = DataModel::getInstance();
  Xmlizer::ToXml("gapphost.conf.xml", model);
#else
  Paper paper;
  paper.setRun(0);
  ArgumentsParser::Parse(paper, argc, argv);
  Xmlizer::FromXml(paper, paper.getConfigurationName());
  paper.setRun(1);
  Xmlizer::ToXml(paper.getConfigurationName(), paper);
#endif
  cublasShutdown();
  return ret;
}
