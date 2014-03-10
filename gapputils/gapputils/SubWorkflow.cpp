#include "SubWorkflow.h"

#include <capputils/attributes/TimeStampAttribute.h>
#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/FlagAttribute.h>

using namespace capputils::attributes;

namespace interfaces {

BeginPropertyDefinitions(SubWorkflow)
  ReflectableBase(gapputils::workflow::WorkflowInterface)

  DefineProperty(Atomic, Flag(), Description("Makes the workflow stateless. Memory will be freed as soon as possible. All modules need to be updated when the workflow needs an update."), Observe(Id))

EndPropertyDefinitions

SubWorkflow::SubWorkflow(void) : _Atomic(false) {
  setLabel("SubWorkflow");
}

}
