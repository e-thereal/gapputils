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
#include <capputils/GenerateBashCompletion.h>
#include <capputils/EnumeratorAttribute.h>
#include <sstream>

#include "DataModel.h"
#include "Workflow.h"
#include "LogbookModel.h"
#include <capputils/Logbook.h>
#include <gapputils/SubWorkflow.h>

#include <memory>

//#include <CProcessInfo.hpp>

using namespace gapputils::host;
using namespace gapputils::workflow;
using namespace gapputils;
using namespace capputils;
using namespace std;
using namespace interfaces;

#include <boost/filesystem.hpp>

#include <algorithm>
#include <exception>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "gapphost.h"

#include <stdio.h>
#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <map>

#include <cstdio> // [s]print[f]

#if defined( WIN32 ) || defined( _WIN32 )
# include <windows.h>
#endif

void showWorkflowUsage(Workflow& workflow) {
  std::cout << "Workflow parameters:\n" << std::endl;

  reflection::IClassProperty *label, *description, *valueProp;

  std::vector<std::string> parameters, descriptions;
  size_t columnWidth = 0;

  workflow.identifyInterfaceNodes();
  std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow.getInterfaceNodes();
  for (size_t i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<Node> node = interfaceNodes[i].lock();
    if (!workflow.isOutputNode(node) && node->getModule()) {
      boost::shared_ptr<reflection::ReflectableClass> module = node->getModule();
      if ((label = module->findProperty("Label"))) {
        columnWidth = max(columnWidth, label->getStringValue(*module).size());
        parameters.push_back(label->getStringValue(*module));

        std::string fullDescription;
        if ((description = module->findProperty("Description")))
          fullDescription = description->getStringValue(*module);

        if ((valueProp = module->findProperty("Value"))) {
          attributes::IEnumeratorAttribute* enumAttr = valueProp->getAttribute<attributes::IEnumeratorAttribute>();
          if (enumAttr) {
            boost::shared_ptr<capputils::AbstractEnumerator> enumerator = enumAttr->getEnumerator(*module, valueProp);
            if (enumerator) {
              std::stringstream valuesDescription;
              if (fullDescription.size())
                valuesDescription << fullDescription << " (";
              else
                valuesDescription << "Possible values are: ";
              vector<string>& values = enumerator->getValues();
              for (unsigned i = 0; i < values.size(); ++i) {
                if (i)
                  valuesDescription << ", ";
                valuesDescription << values[i];
              }
              if (fullDescription.size())
                valuesDescription << ")";
              fullDescription = valuesDescription.str();
            }
          }
        }
        descriptions.push_back(fullDescription);
      }
    }
  }

  assert(parameters.size() == descriptions.size());
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::cout << "  --" << std::setw(columnWidth) << std::left << parameters[i] << "   " << descriptions[i] << std::endl;
  }
  std::cout << std::endl;
}

void parseWorkflowParameters(int argc, char** argv, Workflow& workflow) {
  std::map<std::string, reflection::ReflectableClass*> modules;
  std::map<std::string, reflection::IClassProperty*> properties;

  reflection::IClassProperty *label, *value;

  workflow.identifyInterfaceNodes();
  std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow.getInterfaceNodes();
  for (size_t i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<Node> node = interfaceNodes[i].lock();
    if (!workflow.isOutputNode(node) && node->getModule()) {
      boost::shared_ptr<reflection::ReflectableClass> module = node->getModule();
      if ((label = module->findProperty("Label")) && (value = module->findProperty("Value"))) {
        modules[label->getStringValue(*module)] = module.get();
        properties[label->getStringValue(*module)] = value;
      }
    }
  }

  for (int i = 0; i < argc; ++i) {
    if (!strncmp(argv[i], "--", 2)) {
      std::string propName = argv[i] + 2;
      if (modules.find(propName) != modules.end() && properties.find(propName) != properties.end()) {
        if (i < argc - 1)
          properties[propName]->setStringValue(*modules[propName], argv[++i]);
      }
    }
  }
}

void listWorkflowParameters(Workflow& workflow) {
  reflection::IClassProperty *label;
  workflow.identifyInterfaceNodes();
  std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow.getInterfaceNodes();
  for (size_t i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<Node> node = interfaceNodes[i].lock();
    if (!workflow.isOutputNode(node) && node->getModule()) {
      boost::shared_ptr<reflection::ReflectableClass> module = node->getModule();
      if ((label = module->findProperty("Label"))) {
        std::cout << "--" << label->getStringValue(*module) << " ";
      }
    }
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[])
{
  Logbook dlog(&LogbookModel::GetInstance());
  dlog.setModule("host");

  qRegisterMetaType<std::string>("std::string");

  QCoreApplication::setOrganizationName("gapputils");
  QCoreApplication::setOrganizationDomain("gapputils.blogspot.com");
  QCoreApplication::setApplicationName("grapevine");

  cublasInit();

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
    boost::shared_ptr<Workflow> workflow(new Workflow());
    model.setMainWorkflow(workflow);
    //model.setCurrentWorkflow(workflow->getUuid());
  }
  if (!model.getMainWorkflow()->getModule())
    model.getMainWorkflow()->setModule(boost::shared_ptr<SubWorkflow>(new SubWorkflow()));

  reflection::ReflectableClass& wfModule = *model.getMainWorkflow()->getModule();

  ArgumentsParser::Parse(model, argc, argv);    // Needs to be here again to override configuration file parameters
//  ArgumentsParser::Parse(wfModule, argc, argv);
  parseWorkflowParameters(argc, argv, *model.getMainWorkflow());
  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage("gapphost", model);
    showWorkflowUsage(*model.getMainWorkflow());

    cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
    culaShutdown();
#endif
    return 0;
  }

  if (model.getWorkflowParameters()) {
    listWorkflowParameters(*model.getMainWorkflow());
    return 0;
  }

  if (model.getGenerateBashCompletion().size()) {
    GenerateBashCompletion::Generate(argv[0], model, model.getGenerateBashCompletion());

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
    if (model.getRun()) {
      w.setAutoQuit(true);
      w.updateMainWorkflow();
    }
    ret = a.exec();
  } catch (char const* error) {
    cout << error << endl;
    return 1;
  }

  if (model.getSaveConfiguration())
    model.save();
  model.setMainWorkflow(boost::shared_ptr<Workflow>());

#if !defined( WIN32 ) && !defined( _WIN32 )
  if (model.getEmailLog().size()) {
    std::stringstream mailCommand;
    mailCommand << "cat " << model.getLogfileName() << " | mutt -s \"Grapevine logfile\" -a \"" << model.getConfiguration() << "\" " << model.getEmailLog();
    dlog() << "Executing: " << mailCommand.str() << std::endl;
    system(mailCommand.str().c_str());
  }
#endif

  cublasShutdown();
#ifdef GAPPHOST_CULA_SUPPORT
  culaShutdown();
#endif
  std::cout << "Good bye." << std::endl;

  return ret;
}
