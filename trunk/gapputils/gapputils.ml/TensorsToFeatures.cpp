/*
 * TensorsToFeatures.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#include "TensorsToFeatures.h"

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
#include <capputils/NoParameterAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int TensorsToFeatures::inputId;

BeginPropertyDefinitions(TensorsToFeatures)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Tensors, Input(""), Volatile(), ReadOnly(), Observe(inputId = Id))
  DefineProperty(Width, NoParameter(), Observe(Id))
  DefineProperty(Height, NoParameter(), Observe(Id))
  DefineProperty(Depth, NoParameter(), Observe(Id))
  DefineProperty(Features, Output(""), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Auto, Observe(Id))

EndPropertyDefinitions

TensorsToFeatures::TensorsToFeatures() : _Width(0), _Height(0), _Depth(0), _Auto(false), data(0) {
  WfeUpdateTimestamp
  setLabel("T2F");

  Changed.connect(capputils::EventHandler<TensorsToFeatures>(this, &TensorsToFeatures::changedHandler));
}

TensorsToFeatures::~TensorsToFeatures() {
  if (data)
    delete data;
}

void TensorsToFeatures::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
    writeResults();
  }
}

void TensorsToFeatures::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new TensorsToFeatures();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getTensors() || getTensors()->size() == 0) {
    return;
  }

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getTensors();
  const unsigned count = tensors[0]->data().size();
  boost::shared_ptr<std::vector<value_t> > features(new std::vector<value_t>(count * tensors.size()));

  for (unsigned i = 0; i < tensors.size(); ++i) {
    thrust::copy(tensors[i]->begin(), tensors[i]->end(), features->begin() + i * count);
    if (monitor) monitor->reportProgress(i * 100 / tensors.size());
  }

  data->setWidth(tensors[0]->size()[0]);
  data->setHeight(tensors[0]->size()[1]);
  data->setDepth(tensors[0]->size()[2]);
  data->setFeatures(features);
}

void TensorsToFeatures::writeResults() {
  if (!data)
    return;

  setWidth(data->getWidth());
  setHeight(data->getHeight());
  setDepth(data->getDepth());
  setFeatures(data->getFeatures());
}

}

}
