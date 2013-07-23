/*
 * Sampler.cpp
 *
 *  Created on: Jul 15, 2013
 *      Author: tombr
 */

#include "Sampler.h"

#include <capputils/FlagAttribute.h>
#include <capputils/DeprecatedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Sampler, Deprecated("Use gml.dbm.Sampler instead."))

  ReflectableBase(DefaultWorkflowElement<Sampler>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>())
  WorkflowProperty(GpuCount)
  WorkflowProperty(SampleCount)
  WorkflowProperty(Iterations)
  WorkflowProperty(Damped, Flag())
  WorkflowProperty(Samples, Output("Ts"))

EndPropertyDefinitions

Sampler::Sampler() : _GpuCount(1), _SampleCount(1), _Iterations(1), _Damped(false) {
  setLabel("Sample");
}

SamplerChecker samplerChecker;

} /* namespace convrbm4d */

} /* namespace gml */
