#include "DataModel.h"

#include <ReuseAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)

  ReflectableProperty(MainWorkflow)

EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void)
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
  
}

}
