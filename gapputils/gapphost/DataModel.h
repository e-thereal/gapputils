#pragma once

#ifndef _GAPPHOST_DATAMODEL_H_
#define _GAPPHOST_DATAMODEL_H_

#include <capputils/ReflectableClass.h>

#include <map>

#include "Workflow.h"

class QLabel;

namespace gapputils {

namespace host {

class MainWindow;

class DataModel : public capputils::reflection::ReflectableClass
{
  InitReflectableClass(DataModel)

  Property(Run, bool)
  Property(Help, bool)
  Property(AutoReload, bool)
  Property(WindowX, int)
  Property(WindowY, int)
  Property(WindowWidth, int)
  Property(WindowHeight, int)
  Property(MainWorkflow, workflow::Workflow*)
  Property(OpenWorkflows, boost::shared_ptr<std::vector<std::string> >)
  Property(CurrentWorkflow, std::string)
  Property(WorkflowMap, boost::shared_ptr<std::map<std::string CAPPUTILS_COMMA() workflow::Workflow*> >)     ///< Getter and setters only. No DefineProperty in the cpp file
  Property(MainWindow, MainWindow*)
  Property(PassedLabel, QLabel*)
  Property(RemainingLabel, QLabel*)
  Property(TotalLabel, QLabel*)
  Property(FinishedLabel, QLabel*)
  Property(Configuration, std::string)

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
