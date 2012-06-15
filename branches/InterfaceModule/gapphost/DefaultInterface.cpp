#include "DefaultInterface.h"

#include <capputils/TimeStampAttribute.h>

namespace gapputils {

namespace host {

BeginPropertyDefinitions(DefaultInterface)
  ReflectableBase(gapputils::workflow::WorkflowInterface)
EndPropertyDefinitions

DefaultInterface::DefaultInterface(void)
{
  WfiUpdateTimestamp
  setLabel("DefaultInterface");
}

DefaultInterface::~DefaultInterface(void)
{
}

}

}
