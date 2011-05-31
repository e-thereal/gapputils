#include "Interface.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/TimeStampAttribute.h>

using namespace capputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(Interface)

  ReflectableBase(gapputils::workflow::WorkflowInterface)

  DefineProperty(FirstColumn, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LastColumn, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(GoalColumn, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FirstTrainRow, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LastTrainRow, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Train, Input(), Observe(PROPERTY_ID), Filename(), TimeStamp(PROPERTY_ID))
  DefineProperty(FirstTestRow, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LastTestRow, Input(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Test, Input(), Observe(PROPERTY_ID), Filename(), TimeStamp(PROPERTY_ID))
  DefineProperty(Error, Output(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Interface::Interface(void) : _FirstColumn(1), _LastColumn(-1), _GoalColumn(0),
_FirstTrainRow(1), _LastTrainRow(-1), _Train(""), _FirstTestRow(1), _LastTestRow(-1),
_Test("")
{
  WfiUpdateTimestamp
  setLabel("Interface");
}


Interface::~Interface(void)
{
}

}
