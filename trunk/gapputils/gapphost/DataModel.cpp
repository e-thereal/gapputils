#include "DataModel.h"

#include <ReuseAttribute.h>
#include <VolatileAttribute.h>
#include <FlagAttribute.h>
#include <tinyxml.h>
#include <Xmlizer.h>

using namespace capputils;
using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)
  DefineProperty(NoGui, Flag(), Volatile())
  DefineProperty(Help, Flag(), Volatile())
  DefineProperty(AutoReload, Flag(), Volatile())
  ReflectableProperty(MainWorkflow, Volatile())

EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _NoGui(false), _Help(false), _AutoReload(false), _MainWorkflow(0)
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
  TiXmlElement* modelElement = Xmlizer::CreateXml(*this);
  TiXmlElement* mainWorkflowElement = new TiXmlElement("MainWorkflow");
  TiXmlElement* workflowElement = getMainWorkflow()->getXml(false);
  Xmlizer::ToXml(*workflowElement, *getMainWorkflow());

  mainWorkflowElement->LinkEndChild(workflowElement);
  modelElement->LinkEndChild(mainWorkflowElement);

  Xmlizer::ToFile(filename, modelElement);
}
  
}

}
