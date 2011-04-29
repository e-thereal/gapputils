#include "DataModel.h"

#include <ReuseAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace workflow;

namespace host {

BeginPropertyDefinitions(DataModel)

  ReflectableProperty(Graph, Reuse())

EndPropertyDefinitions

DataModel* DataModel::instance = 0;

DataModel::DataModel(void)
{
  _Graph = new Graph();
}

DataModel::~DataModel(void)
{
  delete _Graph;
}

DataModel& DataModel::getInstance() {
  if (!instance)
    instance = new DataModel();
  return *instance;
}
  
}

}
