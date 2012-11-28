#include "DefaultInterface.h"

#include <capputils/TimeStampAttribute.h>
#include <capputils/DeprecatedAttribute.h>

namespace gapputils {

namespace host {

BeginPropertyDefinitions(DefaultInterface, capputils::attributes::Deprecated("Use 'interfaces::SubWorkflow' instead."))
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
