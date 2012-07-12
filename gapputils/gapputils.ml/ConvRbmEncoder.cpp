/*
 * ConvRbmEncoder.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#include "ConvRbmEncoder.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NoParameterAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

DefineEnum(CodingDirection);
DefineEnum(PoolingMethod);

int ConvRbmEncoder::inputId;

BeginPropertyDefinitions(ConvRbmEncoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Model, Input("CRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Inputs, Input("X"), Volatile(), ReadOnly(), Observe(inputId = PROPERTY_ID))
  DefineProperty(Outputs, Output("Y"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  ReflectableProperty(Direction, Observe(PROPERTY_ID))
  DefineProperty(Sampling, Observe(PROPERTY_ID))
  ReflectableProperty(Pooling, Observe(PROPERTY_ID))
  DefineProperty(Auto, Observe(PROPERTY_ID))
  DefineProperty(OutputDimension, NoParameter(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ConvRbmEncoder::ConvRbmEncoder() : _Sampling(false), _Pooling(PoolingMethod::NoPooling), _Auto(false) {
  WfeUpdateTimestamp
  setLabel("ConvRbmEncoder");

  Changed.connect(capputils::EventHandler<ConvRbmEncoder>(this, &ConvRbmEncoder::changedHandler));
}

ConvRbmEncoder::~ConvRbmEncoder() { }

void ConvRbmEncoder::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
    writeResults();
  }
}

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl
#define LOCATE2(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

/*
 * Model, boost::shared_ptr<ConvRbmModel>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(Sampling, bool)
  Property(Pooling, PoolingMethod)
  Property(Auto, bool)
  Property(OutputDimension
 */

}

}
