#include "MainWindow.h"
#include <QtGui/QApplication>

// TODO: do the cuda, cublas and cula initialization stuff only if requested
#include <cublas.h>
#ifdef GAPPHOST_CULA_SUPPORT
#include <cula.h>
#endif

#include <capputils/Xmlizer.h>
#include <capputils/ArgumentsParser.h>
#include <capputils/Verifier.h>
#include <iostream>
#include <capputils/ReflectableClassFactory.h>
#include <capputils/FactoryException.h>

//#include "../GaussianProcesses/Paper.h"
#include "DataModel.h"
#include "Workflow.h"
#include "DefaultInterface.h"

#include <CProcessInfo.hpp>

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;

int main(int argc, char *argv[])
{
  cublasInit();
#ifdef GAPPHOST_CULA_SUPPORT
  culaStatus status;

  if ((status = culaInitialize()) != culaNoError) {
    std::cout << "Could not initialize CULA: " << culaGetStatusString(status) << std::endl;
    return 1;
  }
#endif

  MSMRI::CProcessInfo::getInstance().getCommandLine(argc, argv);

  int ret = 0;
  QApplication a(argc, argv);
  DataModel& model = DataModel::getInstance();
  try {
  Xmlizer::FromXml(model, "gapphost.conf.xml");
  } catch (capputils::exceptions::FactoryException ex) {
    cout << ex.what() << endl;
    return 1;
  }
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

    cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
    culaShutdown();
#endif
    return 0;
  }

  model.getMainWorkflow()->resumeFromModel();

  try {
    MainWindow w;
    if (model.getNoGui()) {
      // TODO: wait until update has finished.
      model.getMainWorkflow()->updateOutputs();
    } else {
      w.show();
      model.getMainWorkflow()->resumeViewport();
      ret = a.exec();
    }
  } catch (char const* error) {
    cout << error << endl;
    return 1;
  }

  model.saveToFile("gapphost.conf.xml");
  delete model.getMainWorkflow();

  cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
  culaShutdown();
#endif
  return ret;
}
