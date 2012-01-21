#define BOOST_FILESYSTEM_VERSION 2

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

#include "DataModel.h"
#include "Workflow.h"
#include "DefaultInterface.h"

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;

#include <culib/lintrans.h>
#include <boost/filesystem.hpp>

#include <algorithm>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

template<class T>
void printIt(const T& x) {
  std:: cout << x << " ";
}

int main(int argc, char *argv[])
{
  cublasInit();

  boost::filesystem::create_directories(".gapphost");
#ifdef GAPPHOST_CULA_SUPPORT
  culaStatus status;

  if ((status = culaInitialize()) != culaNoError) {
    std::cout << "Could not initialize CULA: " << culaGetStatusString(status) << std::endl;
    return 1;
  }
#endif

  int ret = 0;
  QApplication a(argc, argv);
  DataModel& model = DataModel::getInstance();
  try {
    Xmlizer::FromXml(model, "gapphost.conf.xml"); // compatibility to old versions
    Xmlizer::FromXml(model, ".gapphost/config.xml");
  } catch (capputils::exceptions::FactoryException ex) {
    cout << ex.what() << endl;
    return 1;
  }

  // Initialize if necessary
  if (!model.getMainWorkflow()) {
    Workflow* workflow = new Workflow();
    model.setMainWorkflow(workflow);
    //model.setCurrentWorkflow(workflow->getUuid());
  }
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

  try {
    MainWindow w;
    if (model.getNoGui()) {
      // TODO: wait until update has finished.
      model.getMainWorkflow()->resume();
      model.getMainWorkflow()->updateOutputs();
    } else {
      w.show();
      w.resume();
      ret = a.exec();
    }
  } catch (char const* error) {
    cout << error << endl;
    return 1;
  }

  model.save();
  delete model.getMainWorkflow();

  cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
  culaShutdown();
#endif
  return ret;
}
