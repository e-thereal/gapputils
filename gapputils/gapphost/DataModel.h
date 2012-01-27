#pragma once

#ifndef _GAPPHOST_DATAMODEL_H_
#define _GAPPHOST_DATAMODEL_H_

#include <capputils/ReflectableClass.h>

#include <map>

#include "Workflow.h"

namespace gapputils {

namespace host {

class BuilderSettings : public capputils::reflection::ReflectableClass {
  InitReflectableClass(BuilderSettings)

  Property(CompilerName, std::string)
  Property(IncludeSwitch, std::string)
  Property(OutputSwitch, std::string)
  Property(IncludeDirectories, boost::shared_ptr<std::vector<std::string> >)
  Property(CompilerFlags, boost::shared_ptr<std::vector<std::string> >)

public:
  BuilderSettings();

  virtual ~BuilderSettings();
};

class XsltSettings : public capputils::reflection::ReflectableClass {
  InitReflectableClass(XsltSettings)

  Property(CombinerInterfaceStyleSheetName, std::string)
  Property(StandardInterfaceStyleSheetName, std::string)
  Property(CommandName, std::string)
  Property(InputSwitch, std::string)
  Property(OutputSwitch, std::string)
  Property(XsltSwitch, std::string)

public:
  XsltSettings();
  virtual ~XsltSettings();
};

class DataModel : public capputils::reflection::ReflectableClass
{
  InitReflectableClass(DataModel)

  Property(NoGui, bool)
  Property(Help, bool)
  Property(AutoReload, bool)
  Property(WindowX, int)
  Property(WindowY, int)
  Property(WindowWidth, int)
  Property(WindowHeight, int)
  Property(BuilderSettings, BuilderSettings*)
  Property(XsltSettings, XsltSettings*)
  Property(MainWorkflow, workflow::Workflow*)
  Property(OpenWorkflows, boost::shared_ptr<std::vector<std::string> >)
  Property(CurrentWorkflow, std::string)
  Property(WorkflowMap, boost::shared_ptr<std::map<std::string CAPPUTILS_COMMA() workflow::Workflow*> >)     ///< Getter and setters only. No DefineProperty in the cpp file


private:
  static DataModel* instance;

protected:
  DataModel(void);

public:
  virtual ~DataModel(void);

  static DataModel& getInstance();

  void save();

  static std::string getConfigurationDirectory();
};

}

}

#endif
