/*
 * Sampler.cpp
 *
 *  Created on: Jul 15, 2013
 *      Author: tombr
 */

#include "Sampler.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Sampler)

  ReflectableBase(DefaultWorkflowElement<Sampler>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>())
  WorkflowProperty(SampleCount)
  WorkflowProperty(Iterations)
  WorkflowProperty(GpuCount)
  WorkflowProperty(Samples, Output("Ts"))

EndPropertyDefinitions

Sampler::Sampler() : _SampleCount(1), _Iterations(1), _GpuCount(1) {
  setLabel("Sample");
}

SamplerChecker samplerChecker;

} /* namespace convrbm4d */

} /* namespace gml */
