/*
 * Compare.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: tombr
 */

#include "Compare.h"

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Compare)

  ReflectableBase(DefaultWorkflowElement<Compare>)

  WorkflowProperty(Image1, Input("I1"), NotNull<Type>())
  WorkflowProperty(Image2, Input("I2"), NotNull<Type>())
  WorkflowProperty(Measure, Enumerator<Type>())
  WorkflowProperty(Value, Output(""))

EndPropertyDefinitions

Compare::Compare() {
  setLabel("MSE");
}

void Compare::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  image_t& input1 = *getImage1();
  image_t& input2 = *getImage2();

  tensor<float, 3> img1(input1.getSize()[0], input1.getSize()[1], input1.getSize()[2]);
  tensor<float, 3> img2(input2.getSize()[0], input2.getSize()[1], input2.getSize()[2]);

  thrust::copy(input1.begin(), input1.end(), img1.begin());
  thrust::copy(input2.begin(), input2.end(), img2.begin());

  switch (getMeasure()) {
  case SimilarityMeasure::MSE:
    img1 = (img1 - img2) * (img1 - img2);
    newState->setValue(sum(img1) / img1.count());
    return;

  default:
    dlog(Severity::Warning) << "Unsupported measure '" << getMeasure() << "'. Aborting!";
    return;
  }
}

} /* namespace imageprocessing */

} /* namespace gml */
