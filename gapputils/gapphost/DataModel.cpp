#include "DataModel.h"

#include <capputils/ReuseAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/FlagAttribute.h>
#include <tinyxml/tinyxml.h>
#include <capputils/Xmlizer.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ReuseAttribute.h>
#include <capputils/DescriptionAttribute.h>

#include <qdir.h>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)
  DefineProperty(Run, Flag(), Volatile(),
      Description("Automatically update the workflow and quit aftwards"))
  DefineProperty(Help, Flag(), Volatile(),
      Description("Shows this help"))
  DefineProperty(AutoReload, Flag(), Volatile(),
      Description("Automatically reloads the workflow if one of the loaded libraries has been changend"))
  DefineProperty(WindowX,
      Description("X position of the top left corner of the main window"))
  DefineProperty(WindowY,
      Description("Y position of the top left corner of the main window"))
  DefineProperty(WindowWidth,
      Description("Width of the main window"))
  DefineProperty(WindowHeight,
      Description("Height of the main window"))
  ReflectableProperty(MainWorkflow,
      Description(""))
  DefineProperty(OpenWorkflows, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>(),
      Description(""))
  DefineProperty(CurrentWorkflow,
      Description(""))
  DefineProperty(MainWindow, Volatile(),
      Description(""))
  DefineProperty(PassedLabel, Volatile())
  DefineProperty(RemainingLabel, Volatile())
  DefineProperty(TotalLabel, Volatile())
  DefineProperty(FinishedLabel, Volatile())
  DefineProperty(Configuration, Volatile(),
      Description("Name of the workflow configuration file"))
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _Run(false), _Help(false), _AutoReload(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600),
    _OpenWorkflows(new std::vector<std::string>()),
    _WorkflowMap(new std::map<std::string, boost::shared_ptr<workflow::Workflow> >),
    _Configuration(".gapphost/config.xml")
{
}

DataModel::~DataModel(void)
{
}

DataModel& DataModel::getInstance() {
  if (!instance)
    instance = new DataModel();
  return *instance;
}

void DataModel::save() const {
  save(getConfiguration());
}

void DataModel::save(const std::string& filename) const {
  Xmlizer::ToXml(filename, *this);
}

std::string DataModel::getConfigurationDirectory() {
  return std::string((QDir::homePath() + "/.gapphost").toAscii().data());
}
  
}

}
