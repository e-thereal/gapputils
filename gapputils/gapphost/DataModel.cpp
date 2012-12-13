#include "DataModel.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/ReuseAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/FlagAttribute.h>
#include <tinyxml/tinyxml.h>
#include <capputils/Xmlizer.h>
#include <capputils/EnumerableAttribute.h>
#include <capputils/ReuseAttribute.h>
#include <capputils/DescriptionAttribute.h>
#include <capputils/FilenameAttribute.h>

#include <cstdlib>

#include <qdir.h>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

int DataModel::WorkflowMapId = -1;

BeginPropertyDefinitions(DataModel)
  DefineProperty(Run, Flag(), Volatile(),
      Description("Automatically update the workflow and quit aftwards"))
  DefineProperty(Headless, Flag(), Volatile(),
      Description("Starts grapevine without showing the main window (but you still need X). This really only makes sense along with the Run flag, hence Run is automatically activated."))
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
  DefineProperty(WorkflowMap, Volatile(), Observe(WorkflowMapId = Id),
      Description(""))
  DefineProperty(MainWindow, Volatile(),
      Description(""))
  DefineProperty(PassedLabel, Volatile())
  DefineProperty(RemainingLabel, Volatile())
  DefineProperty(TotalLabel, Volatile())
  DefineProperty(FinishedLabel, Volatile())
  DefineProperty(Configuration, Volatile(), Filename(),
      Description("Name of the workflow configuration file"))
  DefineProperty(LibraryPath, Volatile(), Filename(),
      Description("Path where default libraries are searched. The default value is read from 'GRAPEVINE_LIBRARY_PATH'"))
  DefineProperty(LogfileName, Volatile(), Filename(),
      Description("Name of the file to which log messages will be written. Default is 'grapevine.log'."))
  DefineProperty(SaveConfiguration, Volatile(),
      Description("If set to true, the current configuration is saved when the program exists. Default is true."))
  DefineProperty(EmailLog, Volatile(),
      Description("If set, the final logfile will be send by e-mail to the given address."))
  DefineProperty(GenerateBashCompletion, Volatile(), Filename(),
      Description("Generates a bash_completion configuration file for grapevine."))
  DefineProperty(WorkflowParameters, Volatile(), Flag(),
      Description("Returns the list of workflow parameters."))
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _Run(false), _Headless(false), _Help(false), _AutoReload(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600),
    _OpenWorkflows(new std::vector<std::string>()),
    _WorkflowMap(new std::map<std::string, boost::weak_ptr<workflow::Workflow> >),
    _Configuration(".gapphost/config.xml"), _LogfileName("grapevine.log"), _SaveConfiguration(true),
    _WorkflowParameters(false)
{
#if defined(_RELEASE)
  char* path = std::getenv("GRAPEVINE_LIBRARY_PATH");
#elif defined(_DEBUG)
  char* path = std::getenv("GRAPEVINE_DEBUG_LIBRARY_PATH");
#else
#error "Neither _DEBUG nor _RELEASE has been defined."
#endif
  if (path)
    setLibraryPath(path);
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
