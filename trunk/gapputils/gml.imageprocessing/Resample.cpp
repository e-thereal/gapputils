/*
 * Resample.cpp
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#include "Resample.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Resample)

  ReflectableBase(DefaultWorkflowElement<Resample>)

  WorkflowProperty(Input, Input("I"), NotNull<Type>())
  WorkflowProperty(Size)
  WorkflowProperty(PixelSize)
  WorkflowProperty(Output, Output("I"))

EndPropertyDefinitions

Resample::Resample() : _Size(tbblas::seq<3>(1)), _PixelSize(tbblas::seq<3>(1000)) {
  setLabel("Resample");
}

ResampleChecker resampleChecker;

} /* namespace imageprocessing */

} /* namespace gml */
