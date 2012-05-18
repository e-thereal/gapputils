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

BeginPropertyDefinitions(BuilderSettings)

  DefineProperty(CompilerName,
      Description("Name of the compiler (e.g. 'gcc', 'cl')"))
  DefineProperty(IncludeSwitch,
      Description("Name of the switch used to add a new include director (e.g. '-I')"))
  DefineProperty(OutputSwitch,
      Description("Name of the switch used to specify the output name. (e.g. '-o ')"))
  DefineProperty(IncludeDirectories,
      Enumerable<TYPE_OF(IncludeDirectories), false>(),
      Description("List of additional include directories"))
  DefineProperty(CompilerFlags,
      Enumerable<TYPE_OF(CompilerFlags), false>(),
      Description("List of additional compiler flags"))

EndPropertyDefinitions

BuilderSettings::BuilderSettings()
 : _CompilerName("gcc"),
   _IncludeSwitch("-I"),
   _OutputSwitch("-o ")
{
  boost::shared_ptr<std::vector<std::string> > includeDirectories(new std::vector<std::string>());
  includeDirectories->push_back("/home/tombr/Projects");
  includeDirectories->push_back("/home/tombr/include");
  includeDirectories->push_back("/home/tombr/Programs/cuda/include");
  includeDirectories->push_back("/home/tombr/Programs/Qt-4.7.3/include");
  includeDirectories->push_back("/home/tombr/Programs/Qt-4.7.3/include/QtCore");
  includeDirectories->push_back("/home/tombr/Programs/Qt-4.7.3/include/QtGui");
  setIncludeDirectories(includeDirectories);

  boost::shared_ptr<std::vector<std::string> > compilerFlags(new std::vector<std::string>());
  compilerFlags->push_back("-shared");
  compilerFlags->push_back("-std=c++0x");
  setCompilerFlags(compilerFlags);
}

BuilderSettings::~BuilderSettings() { }

BeginPropertyDefinitions(XsltSettings)

  DefineProperty(CombinerInterfaceStyleSheetName)
  DefineProperty(StandardInterfaceStyleSheetName)
  DefineProperty(CommandName)
  DefineProperty(InputSwitch)
  DefineProperty(OutputSwitch)
  DefineProperty(XsltSwitch)

EndPropertyDefinitions

XsltSettings::XsltSettings()
 : _CombinerInterfaceStyleSheetName("combiner_interface.xslt"),
   _StandardInterfaceStyleSheetName("standard_interface.xslt"),
#ifdef _WIN32
   _CommandName("altovaxml"),
   _InputSwitch("-in "),
   _OutputSwitch("-out "),
   _XsltSwitch("-xslt2 ")
#else
   _CommandName("java -jar ~/tools/saxon9he.jar"),
   _InputSwitch("-s:"),
   _OutputSwitch("-o:"),
   _XsltSwitch("-xsl:")
#endif
{
}


XsltSettings::~XsltSettings() { }

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
  ReflectableProperty(BuilderSettings, Reuse(),
      Description(""))
  ReflectableProperty(XsltSettings, Reuse(),
      Description(""))
  ReflectableProperty(MainWorkflow,
      Description(""))
  DefineProperty(OpenWorkflows, Enumerable<boost::shared_ptr<std::vector<std::string> >, false>(),
      Description(""))
  DefineProperty(CurrentWorkflow,
      Description(""))
  DefineProperty(MainWindow, Volatile(),
      Description(""))
  DefineProperty(Configuration, Volatile(),
      Description("Name of the workflow configuration file"))
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _Run(false), _Help(false), _AutoReload(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600),
    _BuilderSettings(new BuilderSettings()),
    _XsltSettings(new XsltSettings()),
    _MainWorkflow(0),
    _OpenWorkflows(new std::vector<std::string>()),
    _WorkflowMap(new std::map<std::string, workflow::Workflow*>),
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
  //TiXmlElement* modelElement = Xmlizer::CreateXml(*this);
  //TiXmlElement* mainWorkflowElement = new TiXmlElement("MainWorkflow");
  //TiXmlElement* workflowElement = getMainWorkflow()->getXml(false);
  //Xmlizer::ToXml(*workflowElement, *getMainWorkflow());

  //mainWorkflowElement->LinkEndChild(workflowElement);
  //modelElement->LinkEndChild(mainWorkflowElement);

  //Xmlizer::ToFile(filename, modelElement);

  Xmlizer::ToXml(filename, *this);
}

std::string DataModel::getConfigurationDirectory() {
  return std::string((QDir::homePath() + "/.gapphost").toAscii().data());
}
  
}

}
