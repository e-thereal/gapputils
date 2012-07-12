#define BOOST_FILESYSTEM_VERSION 2

#include "MainWindow.h"
#include <QtGui/QApplication>
#include <qdir.h>

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
#include <sstream>

#include "DataModel.h"
#include "Workflow.h"
#include "DefaultInterface.h"
#include "LogbookModel.h"
#include <gapputils/Logbook.h>

//#include <CProcessInfo.hpp>

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;

#include <culib/lintrans.h>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <exception>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "TestThread.h"

#include "gapphost.h"

template<class T>
void printIt(const T& x) {
  std:: cout << x << " ";
}

int main(int argc, char *argv[])
{
  qRegisterMetaType<std::string>("std::string");

  QCoreApplication::setOrganizationName("gapputils");
  QCoreApplication::setOrganizationDomain("gapputils.blogspot.com");
  QCoreApplication::setApplicationName("grapevine");

  cublasInit();
  gapputils::Logbook dlog(&gapputils::host::LogbookModel::GetInstance());
  dlog.setModule("host");

  //MSMRI::CProcessInfo::getInstance().getCommandLine(argc, argv);

  boost::filesystem::create_directories(".gapphost");
  boost::filesystem::create_directories(DataModel::getConfigurationDirectory());

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
  ArgumentsParser::Parse(model, argc, argv);      // need to be here to read the configuration filename
  try {
    Xmlizer::FromXml(model, DataModel::getConfigurationDirectory() + "/config.xml");
    Xmlizer::FromXml(model, "gapphost.conf.xml"); // compatibility to old versions
    Xmlizer::FromXml(model, model.getConfiguration());
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

  ArgumentsParser::Parse(model, argc, argv);    // Needs to be here again to override configuration file parameters
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
    w.show();
    dlog() << "Start resuming ...";
    w.resume();
    //dlog() << "[Info] Resuming done.";
    if (model.getRun()) {
      w.setAutoQuit(true);
      //std::cout << "[Info] Update main workflow." << std::endl;
      w.updateMainWorkflow();
    }
    //std::cout << "[Info] Entering event loop." << std::endl;
    ret = a.exec();
    //std::cout << "[Info] Quitting." << std::endl;
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
