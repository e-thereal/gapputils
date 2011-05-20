#include "Interface.h"

#include <ObserveAttribute.h>
#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <FilenameAttribute.h>

using namespace capputils::attributes;

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
  DefineProperty(Error, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

Interface::Interface(void) : _FirstColumn(1), _LastColumn(-1), _GoalColumn(0),
_FirstTrainRow(1), _LastTrainRow(-1), _Train(""), _FirstTestRow(1), _LastTestRow(-1),
_Test("")
{
}


Interface::~Interface(void)
{
}

}
