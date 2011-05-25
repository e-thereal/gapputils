#include "DataModel.h"

#include <capputils/ReuseAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/FlagAttribute.h>
#include <tinyxml/tinyxml.h>
#include <capputils/Xmlizer.h>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)
  DefineProperty(NoGui, Flag(), Volatile())
  DefineProperty(Help, Flag(), Volatile())
  DefineProperty(AutoReload, Flag(), Volatile())
  DefineProperty(WindowX)
  DefineProperty(WindowY)
  DefineProperty(WindowWidth)
  DefineProperty(WindowHeight)
  ReflectableProperty(MainWorkflow)
EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _NoGui(false), _Help(false), _AutoReload(false),
    _WindowX(150), _WindowY(150), _WindowWidth(1200), _WindowHeight(600), _MainWorkflow(0)
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

void DataModel::saveToFile(const char* filename) {
  //TiXmlElement* modelElement = Xmlizer::CreateXml(*this);
  //TiXmlElement* mainWorkflowElement = new TiXmlElement("MainWorkflow");
  //TiXmlElement* workflowElement = getMainWorkflow()->getXml(false);
  //Xmlizer::ToXml(*workflowElement, *getMainWorkflow());

  //mainWorkflowElement->LinkEndChild(workflowElement);
  //modelElement->LinkEndChild(mainWorkflowElement);

  //Xmlizer::ToFile(filename, modelElement);
  Xmlizer::ToXml(filename, *this);
}
  
}

}
