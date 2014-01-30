#pragma once

#ifndef _GAPPHOST_DATAMODEL_H_
#define _GAPPHOST_DATAMODEL_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>

#include <map>

#include "Workflow.h"
#include <boost/weak_ptr.hpp>

class QLabel;

namespace gapputils {

namespace host {

class MainWindow;

class DataModel : public capputils::reflection::ReflectableClass,
                  public capputils::ObservableClass
{
  InitReflectableClass(DataModel)

  Property(UpdateAll, bool)
  Property(Update, std::vector<std::string>)
  Property(Headless, bool)
  Property(Help, bool)
  Property(HelpAll, bool)
//  Property(AutoReload, bool)
  Property(WindowX, int)
  Property(WindowY, int)
  Property(WindowWidth, int)
  Property(WindowHeight, int)
  Property(MainWorkflow, boost::shared_ptr<workflow::Workflow>)
  Property(OpenWorkflows, boost::shared_ptr<std::vector<std::string> >)
  Property(CurrentWorkflow, std::string)

  // Map is used for opening and closing workflows by UUID
  Property(WorkflowMap, boost::shared_ptr<std::map<std::string CAPPUTILS_COMMA() boost::weak_ptr<workflow::Workflow> > >)
  Property(MainWindow, MainWindow*)
  Property(PassedLabel, QLabel*)
  Property(RemainingLabel, QLabel*)
  Property(TotalLabel, QLabel*)
  Property(FinishedLabel, QLabel*)
  Property(Configuration, std::string)
  Property(LibraryPath, std::string)
  Property(SnippetsPath, std::string)
  Property(LogfileName, std::string)
  Property(SaveConfiguration, bool)
  Property(EmailLog, std::string)
  Property(GenerateBashCompletion, std::string)
  Property(WorkflowParameters, bool)

public:
  static int WorkflowMapId;
  static const char* AutoSaveName;

private:
  static DataModel* instance;

protected:
  DataModel(void);

public:
  virtual ~DataModel(void);

  static DataModel& getInstance();

  void save() const;
  void save(const std::string& filename) const;

  static std::string getConfigurationDirectory();
};

}

}

#endif
