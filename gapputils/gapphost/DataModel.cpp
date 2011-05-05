#include "DataModel.h"

#include <ReuseAttribute.h>
#include <VolatileAttribute.h>
#include <FlagAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)
  DefineProperty(NoGui, Flag(), Volatile())
  DefineProperty(Help, Flag(), Volatile())
  ReflectableProperty(MainWorkflow, Volatile())

EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void) : _NoGui(false), _Help(false)
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
