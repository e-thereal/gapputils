#include "Interface.h"

#include <ObserveAttribute.h>
#include <InputAttribute.h>
#include <FilenameAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(Interface)

  DefineProperty(FirstColumn, Input(), Observe(PROPERTY_ID))
  DefineProperty(LastColumn, Input(), Observe(PROPERTY_ID))
  DefineProperty(GoalColumn, Input(), Observe(PROPERTY_ID))
  DefineProperty(FirstTrainRow, Input(), Observe(PROPERTY_ID))
  DefineProperty(LastTrainRow, Input(), Observe(PROPERTY_ID))
  DefineProperty(Train, Input(), Observe(PROPERTY_ID), Filename())
  DefineProperty(FirstTestRow, Input(), Observe(PROPERTY_ID))
  DefineProperty(LastTestRow, Input(), Observe(PROPERTY_ID))
  DefineProperty(Test, Input(), Observe(PROPERTY_ID), Filename())

EndPropertyDefinitions

Interface::Interface(void)
{
}


Interface::~Interface(void)
{
}

}
