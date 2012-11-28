#include "SubWorkflow.h"

#include <capputils/TimeStampAttribute.h>

namespace interfaces {

BeginPropertyDefinitions(SubWorkflow)
  ReflectableBase(gapputils::workflow::WorkflowInterface)
EndPropertyDefinitions

SubWorkflow::SubWorkflow(void)
{
  setLabel("SubWorkflow");
}

SubWorkflow::~SubWorkflow(void)
{
}

}
