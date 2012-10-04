/*
 * ConvRbmEncoder.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#include "ConvRbmEncoder.h"

#include <capputils/EnumeratorAttribute.h>
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
#include <gapputils/GenerateInterfaceAttribute.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int ConvRbmEncoder::inputId;

BeginPropertyDefinitions(ConvRbmEncoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  WorkflowProperty(Model, Input("CRBM"))
  DefineProperty(Inputs, Input("X"), Volatile(), ReadOnly(), Observe(inputId = Id))
  WorkflowProperty(Outputs, Output("Y"), GenerateInterface("Tensors", "tbblas/tensor.hpp"))
//  WorkflowProperty(Debug1, Output("D1"))
//  WorkflowProperty(Debug2, Output("D2"))
//  WorkflowProperty(Debug3, Output("D3"))
  WorkflowProperty(Direction, Enumerator<CodingDirection>())
  WorkflowProperty(Sampling)
  WorkflowProperty(Pooling, Enumerator<PoolingMethod>())
  WorkflowProperty(Auto)
  WorkflowProperty(SingleFilter, Description("-1 indicates use all filters, otherwise the specified filter is used for the reconstruction."))
  WorkflowProperty(OutputDimension, NoParameter())

EndPropertyDefinitions

ConvRbmEncoder::ConvRbmEncoder()
 : _Sampling(false), _Pooling(PoolingMethod::NoPooling), _Auto(false), _SingleFilter(-1)
{
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
