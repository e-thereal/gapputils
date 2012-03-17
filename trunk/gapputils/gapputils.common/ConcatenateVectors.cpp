/*
 * ConcatenateVectors.cpp
 *
 *  Created on: Mar 12, 2012
 *      Author: tombr
 */

#include "ConcatenateVectors.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(ConcatenateVectors)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(Input1, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Input2, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Dimension1, Observe(PROPERTY_ID))
  DefineProperty(Dimension2, Observe(PROPERTY_ID))
  DefineProperty(Output, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(OutputDimension, NoParameter(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ConcatenateVectors::ConcatenateVectors() : _Dimension1(1), _Dimension2(1), _OutputDimension(2), data(0) {
  WfeUpdateTimestamp
  setLabel("CV");

  Changed.connect(capputils::EventHandler<ConcatenateVectors>(this, &ConcatenateVectors::changedHandler));
}

ConcatenateVectors::~ConcatenateVectors() {
  if (data)
    delete data;
}

void ConcatenateVectors::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ConcatenateVectors::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ConcatenateVectors();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInput1() || !getInput2()) {
    std::cout << "[Warning] No input given." << std::endl;
    return;
  }

  if ((getInput1()->size() % getDimension1()) || (getInput1()->size() % getDimension1())) {
    std::cout << "[Warning] Vectors dimenions do not match vector size." << std::endl;
    return;
  }

  std::vector<float>& input1 = *getInput1();
  std::vector<float>& input2 = *getInput2();
  const int dim1 = getDimension1();
  const int dim2 = getDimension2();
  const unsigned cSample = input1.size() / dim1;

  if (cSample != input2.size() / dim2) {
    std::cout << "[Warning] Vectors counts don't match." << std::endl;
    return;
  }

  const int oDim = dim1 + dim2;
  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(oDim * cSample));

  for (unsigned i = 0; i < cSample; ++i) {
    std::copy(input1.begin() + i * dim1, input1.begin() + (i+1) * dim1,
        output->begin() + i * oDim);
    std::copy(input2.begin() + i * dim2, input2.begin() + (i+1) * dim2,
        output->begin() + i * oDim + dim1);
  }

  data->setOutput(output);
}

void ConcatenateVectors::writeResults() {
  if (!data)
    return;

  setOutput(data->getOutput());
  setOutputDimension(getDimension1() + getDimension2());
}

}

}
