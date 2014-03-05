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
#include <capputils/ParameterAttribute.h>
#include <capputils/OperandAttribute.h>

#include <cstdlib>

#include <qdir.h>

#include <boost/filesystem.hpp>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

int DataModel::WorkflowMapId = -1;
const char* DataModel::AutoSaveName = ".grapevine-autosave.xml";

#if defined(_RELEASE)
  #define GRAPEVINE_LIBRARY_PATH "GRAPEVINE_LIBRARY_PATH"
#elif defined(_DEBUG)
  #define GRAPEVINE_LIBRARY_PATH "GRAPEVINE_DEBUG_LIBRARY_PATH"
#else
  #error "Neither _DEBUG nor _RELEASE has been defined."
#endif

BeginPropertyDefinitions(DataModel)
  DefineProperty(UpdateAll, Flag(), Volatile(),
      Parameter("update_all", "a"),
      Description("Automatically update the main workflow and quit afterwards"))
  DefineProperty(Update, Enumerable<Type, false>(), Volatile(),
      Parameter("update", "u"),
      Description("Automatically update only the specified output nodes and quit afterwards"))
  DefineProperty(Headless, Flag(), Volatile(),
      Parameter("headless", ""),
      Description("Start grapevine without a GUI and update the main workflow if no other update target is given"))
  DefineProperty(Help, Flag(), Volatile(),
      Parameter("help", "h"),
      Description("Shows this help"))
  DefineProperty(HelpAll, Flag(), Volatile(),
      Parameter("help_all", ""),
      Description("Shows help for all parameters including parameters that are meant for internal use only."))
//  DefineProperty(AutoReload, Flag(), Volatile(),
//      Description("Automatically reload the workflow if one of the loaded libraries has been changend"))
  DefineProperty(WindowX,
      Description("X position of the top left corner of the main window"))
  DefineProperty(WindowY,
      Description("Y position of the top left corner of the main window"))
  DefineProperty(WindowWidth,
      Description("Width of the main window"))
  DefineProperty(WindowHeight,
      Description("Height of the main window"))
  ReflectableProperty(MainWorkflow)
  DefineProperty(OpenWorkflows, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>())
  DefineProperty(CurrentWorkflow)
  DefineProperty(WorkflowMap, Volatile(), Observe(WorkflowMapId = Id))
  DefineProperty(MainWindow, Volatile())
  DefineProperty(PassedLabel, Volatile())
  DefineProperty(RemainingLabel, Volatile())
  DefineProperty(TotalLabel, Volatile())
  DefineProperty(FinishedLabel, Volatile())
  DefineProperty(Configuration, Volatile(), Filename(),
      Operand("configuration"),
//      Parameter("config", "c"),
      Description("Name of the workflow configuration file"))
  DefineProperty(LibraryPath, Volatile(), Filename(),
      Parameter("libraries", ""),
      Description("Path where default libraries are searched. The default value is read from '" GRAPEVINE_LIBRARY_PATH "'"))
  DefineProperty(SnippetsPath, Volatile(), Filename(),
      Parameter("snippets", ""),
      Description("Path where workflow snippets are searched. The default value is read from 'GRAPEVINE_SNIPPETS_PATH'"))
  DefineProperty(LogfileName, Volatile(), Filename(),
      Parameter("log", "l"),
      Description("Name of the file to which log messages will be written. Default is 'grapevine.log'."))
  DefineProperty(SaveConfiguration, Volatile(),
      Parameter("save", ""),
      Description("If set to true, the current configuration is saved when the program exists. Default is true."))
  DefineProperty(EmailLog, Volatile(),
      Parameter("email", "e"),
      Description("If set, the final logfile will be send by e-mail to the given address."))
  DefineProperty(GenerateBashCompletion, Volatile(), Filename(),
      Parameter("bash", "b"),
      Description("Generate a bash_completion configuration file for grapevine."))
  DefineProperty(WorkflowParameters, Volatile(), Flag(),
      Parameter("parameters", "p"),
      Description("Return the list of workflow parameters."))
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _UpdateAll(false), _Headless(false), _Help(false), _HelpAll(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600),
    _OpenWorkflows(new std::vector<std::string>()),
    _WorkflowMap(new std::map<std::string, boost::weak_ptr<workflow::Workflow> >),
    _Configuration(".gapphost/config.xml"), _LibraryPath(".libraries"), _SnippetsPath(".snippets"), _LogfileName("grapevine.log"), _SaveConfiguration(true),
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

DataModel::~DataModel(void) {
}

DataModel& DataModel::getInstance() {
  if (!instance)
    instance = new DataModel();
  return *instance;
}

void DataModel::save() const {
  save(getConfiguration());
  boost::filesystem::remove(DataModel::AutoSaveName);
}

void DataModel::save(const std::string& filename) const {
  Xmlizer::ToXml(filename, *this);
}

std::string DataModel::getConfigurationDirectory() {
  return std::string((QDir::homePath() + "/.gapphost").toAscii().data());
}
  
}

}
