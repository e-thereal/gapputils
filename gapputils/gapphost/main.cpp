#define BOOST_FILESYSTEM_VERSION 2

#include "MainWindow.h"
#include <QtGui/QApplication>
#include <qdir.h>

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
#include <qmessagebox.h>

#include "HeadlessApp.h"

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

#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <map>

#include <cstdio>

#if defined( WIN32 ) || defined( _WIN32 )
# include <windows.h>
#endif

void createParameterList(Workflow& workflow, std::vector<capputils::ParameterDescription>& parameters) {
  reflection::IClassProperty *label, *description, *valueProp;

  if (workflow.getModule())
    capputils::ArgumentsParser::CreateParameterList(*workflow.getModule(), false, parameters);

  workflow.identifyInterfaceNodes();
  std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow.getInterfaceNodes();
  for (size_t i = 0; i < interfaceNodes.size(); ++i) {
    boost::shared_ptr<Node> node = interfaceNodes[i].lock();
    if (!workflow.isOutputNode(node) && node->getModule()) {
      boost::shared_ptr<reflection::ReflectableClass> module = node->getModule();

      label = module->findProperty("Label");
      description = module->findProperty("Description");
      valueProp = module->findProperty("Values");
      if (!valueProp)
        valueProp = module->findProperty("Value");

      if (valueProp && label) {
        parameters.push_back(capputils::ParameterDescription(module.get(), valueProp, label->getStringValue(*module), "", (description ? description->getStringValue(*module) : "")));
      }
    }
  }
}

void showWorkflowUsage(Workflow& workflow) {
  std::vector<capputils::ParameterDescription> parameters;
  createParameterList(workflow, parameters);

  if (parameters.size()) {
    ArgumentsParser::PrintUsage("Workflow parameters:", parameters);
  }

  {
    workflow.identifyInterfaceNodes();
    std::vector<boost::weak_ptr<Node> >& interfaceNodes = workflow.getInterfaceNodes();

    reflection::IClassProperty *label, *description;
    std::vector<std::string> parameters, descriptions;
    size_t columnWidth = 0;

    // Show output targets
    for (size_t i = 0; i < interfaceNodes.size(); ++i) {
      boost::shared_ptr<Node> node = interfaceNodes[i].lock();
      if (workflow.isOutputNode(node) && node->getModule()) {
        boost::shared_ptr<reflection::ReflectableClass> module = node->getModule();
        if ((label = module->findProperty("Label"))) {
          columnWidth = max(columnWidth, label->getStringValue(*module).size());
          parameters.push_back(label->getStringValue(*module));

          std::string fullDescription;
          if ((description = module->findProperty("Description")))
            fullDescription = description->getStringValue(*module);
          descriptions.push_back(fullDescription);
        }
      }
    }

    assert(parameters.size() == descriptions.size());

    if (parameters.size()) {
      std::cout << "Workflow targets:\n" << std::endl;
      for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << "  * " << std::setw(columnWidth) << std::left << parameters[i] << "   " << descriptions[i] << std::endl;
      }
      std::cout << std::endl;
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
  QCoreApplication* qapp = 0;

  Logbook dlog(&LogbookModel::GetInstance());
  dlog.setModule("host");

  qRegisterMetaType<std::string>("std::string");

  QCoreApplication::setOrganizationName("gapputils");
  QCoreApplication::setOrganizationDomain("gapputils.blogspot.com");
  QCoreApplication::setApplicationName("grapevine");

  boost::filesystem::create_directories(".gapphost");
  boost::filesystem::create_directories(DataModel::getConfigurationDirectory());

  int ret = 0;
  DataModel& model = DataModel::getInstance();
  ArgumentsParser::Parse(model, argc, argv, true);      // need to be here to read the configuration filename

  if (model.getHeadless()) {
    qapp = new QCoreApplication(argc, argv);
  } else {
    qapp = new QApplication(argc, argv);
  }

  try {
    Xmlizer::FromXml(model, DataModel::getConfigurationDirectory() + "/config.xml");
    Xmlizer::FromXml(model, "gapphost.conf.xml"); // compatibility to old versions

    if (boost::filesystem::exists(DataModel::AutoSaveName) && !model.getHeadless() && QMessageBox::question(0, "Crash detected.", "It seems grapevine didn't exit properly or is still running. Do you want to load the last automatically saved configuration?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
      Xmlizer::FromXml(model, DataModel::AutoSaveName);
    } else {
      Xmlizer::FromXml(model, model.getConfiguration());
    }
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

  {
    // Needs to be here again to override configuration file parameters and because only now we know the complete parameter list
    std::vector<capputils::ParameterDescription> parameters;
    capputils::ArgumentsParser::CreateParameterList(model, false, parameters);
    createParameterList(*model.getMainWorkflow(), parameters);
    capputils::ArgumentsParser::Parse(parameters, argc, argv);
  }

  if (model.getHelp()) {
    ArgumentsParser::PrintDefaultUsage(argv[0], model, true);
    showWorkflowUsage(*model.getMainWorkflow());
    return 0;
  }

  if (model.getHelpAll()) {
    ArgumentsParser::PrintDefaultUsage(argv[0], model, false);
    showWorkflowUsage(*model.getMainWorkflow());
    return 0;
  }

  if (model.getWorkflowParameters()) {
    listWorkflowParameters(*model.getMainWorkflow());
    return 0;
  }

  if (model.getGenerateBashCompletion().size()) {
    GenerateBashCompletion::Generate(argv[0], model, model.getGenerateBashCompletion(), true);
    return 0;
  }

  try {
    if (!model.getHeadless()) {
      MainWindow w;
      w.show();
      w.resume();
      if (model.getUpdateAll()) {
        model.setSaveConfiguration(false);
        w.setAutoQuit(true);
        w.updateMainWorkflow();
      } else if (model.getUpdate().size()) {
        model.setSaveConfiguration(false);
        w.setAutoQuit(true);
        w.updateMainWorkflowNodes(model.getUpdate());
      }
      ret = qapp->exec();
    } else {
      model.setSaveConfiguration(false);
      if (model.getUpdate().size() == 0)
        model.setUpdateAll(true);
      HeadlessApp app;
      app.resume();
      bool success = false;
      if (model.getUpdateAll()) {
        success = app.updateMainWorkflow();
      } else if (model.getUpdate().size()) {
        success = app.updateMainWorkflowNodes(model.getUpdate());
      }
      if (success)
        ret = qapp->exec();
    }
  } catch (char const* error) {
    cout << error << endl;
    model.save(DataModel::AutoSaveName);
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

  if (qapp)
    delete qapp;

  std::cout << "Good bye." << std::endl;

  return ret;
}
