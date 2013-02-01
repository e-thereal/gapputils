#define BOOST_FILESYSTEM_VERSION 2

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
#include <capputils/HideAttribute.h>

#include <cstdlib>

#include <qdir.h>

#include <boost/filesystem.hpp>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

int DataModel::WorkflowMapId = -1;

#if defined(_RELEASE)
  #define GRAPEVINE_LIBRARY_PATH "GRAPEVINE_LIBRARY_PATH"
#elif defined(_DEBUG)
  #define GRAPEVINE_LIBRARY_PATH "GRAPEVINE_DEBUG_LIBRARY_PATH"
#else
  #error "Neither _DEBUG nor _RELEASE has been defined."
#endif

BeginPropertyDefinitions(DataModel)
  DefineProperty(UpdateAll, Flag(), Volatile(),
      Description("Automatically update the main workflow and quit afterwards"))
  DefineProperty(Update, Volatile(),
      Description("Automatically update only the specified output node and quit afterwards"))
  DefineProperty(Headless, Flag(), Volatile(),
      Description("Start grapevine without a GUI and update the main workflow if no other update target is given"))
  DefineProperty(Help, Flag(), Volatile(),
      Description("Show this help"))
  DefineProperty(AutoReload, Flag(), Volatile(),
      Description("Automatically reload the workflow if one of the loaded libraries has been changend"))
  DefineProperty(WindowX,
      Description("X position of the top left corner of the main window"))
  DefineProperty(WindowY,
      Description("Y position of the top left corner of the main window"))
  DefineProperty(WindowWidth,
      Description("Width of the main window"))
  DefineProperty(WindowHeight,
      Description("Height of the main window"))
  ReflectableProperty(MainWorkflow,
      Description(""), Hide())
  DefineProperty(OpenWorkflows, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>(),
      Description(""), Hide())
  DefineProperty(CurrentWorkflow,
      Description(""), Hide())
  DefineProperty(WorkflowMap, Volatile(), Observe(WorkflowMapId = Id),
      Description(""), Hide())
  DefineProperty(MainWindow, Volatile(),
      Description(""), Hide())
  DefineProperty(PassedLabel, Volatile(), Hide())
  DefineProperty(RemainingLabel, Volatile(), Hide())
  DefineProperty(TotalLabel, Volatile(), Hide())
  DefineProperty(FinishedLabel, Volatile(), Hide())
  DefineProperty(Configuration, Volatile(), Filename(),
      Description("Name of the workflow configuration file"))
  DefineProperty(LibraryPath, Volatile(), Filename(),
      Description("Path where default libraries are searched. The default value is read from '" GRAPEVINE_LIBRARY_PATH "'"))
  DefineProperty(SnippetsPath, Volatile(), Filename(),
        Description("Path where workflow snippets are searched. The default value is read from 'GRAPEVINE_SNIPPETS_PATH'"))
  DefineProperty(LogfileName, Volatile(), Filename(),
      Description("Name of the file to which log messages will be written. Default is 'grapevine.log'."))
  DefineProperty(SaveConfiguration, Volatile(),
      Description("If set to true, the current configuration is saved when the program exists. Default is true."))
  DefineProperty(EmailLog, Volatile(),
      Description("If set, the final logfile will be send by e-mail to the given address."))
  DefineProperty(GenerateBashCompletion, Volatile(), Filename(),
      Description("Generate a bash_completion configuration file for grapevine."))
  DefineProperty(WorkflowParameters, Volatile(), Flag(),
      Description("Return the list of workflow parameters."))
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _UpdateAll(false), _Headless(false), _Help(false), _AutoReload(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600),
    _OpenWorkflows(new std::vector<std::string>()),
    _WorkflowMap(new std::map<std::string, boost::weak_ptr<workflow::Workflow> >),
    _Configuration(".gapphost/config.xml"), _SnippetsPath(".snippets"), _LogfileName("grapevine.log"), _SaveConfiguration(true),
    _WorkflowParameters(false)
{
  char* path = std::getenv(GRAPEVINE_LIBRARY_PATH);
  if (path)
    setLibraryPath(path);
  boost::filesystem::create_directories(getLibraryPath());
  char* snippets = std::getenv("GRAPEVINE_SNIPPETS_PATH");
  if (snippets)
    setSnippetsPath(snippets);
  boost::filesystem::create_directories(getSnippetsPath());
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
