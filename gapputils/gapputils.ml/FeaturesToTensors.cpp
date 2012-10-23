/*
 * FeaturesToTensors.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#include "FeaturesToTensors.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FeaturesToTensors)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Features, Input(""), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Width, NotEqual<int>(0), Observe(Id))
  DefineProperty(Height, NotEqual<int>(0), Observe(Id))
  DefineProperty(Depth, NotEqual<int>(0), Observe(Id))
  DefineProperty(Tensors, Output(""), Volatile(), ReadOnly(), Observe(Id))

EndPropertyDefinitions

FeaturesToTensors::FeaturesToTensors() : _Width(0), _Height(0), _Depth(1), data(0) {
  WfeUpdateTimestamp
  setLabel("F2T");

  Changed.connect(capputils::EventHandler<FeaturesToTensors>(this, &FeaturesToTensors::changedHandler));
}

FeaturesToTensors::~FeaturesToTensors() {
  if (data)
    delete data;
}

void FeaturesToTensors::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FeaturesToTensors::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FeaturesToTensors();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int width = getWidth();
  const int height = getHeight();
  const int depth = getDepth();
  const int count = width * height * depth;

  if (!getFeatures() || (getFeatures()->size() % count != 0)) {
    return;
  }

  std::vector<value_t>& features = *getFeatures();
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > tensors(new std::vector<boost::shared_ptr<tensor_t> >());

  for (unsigned i = 0; i < features.size(); i += count) {
    boost::shared_ptr<tensor_t> tensor(new tensor_t(width, height, depth));
    thrust::copy(features.begin() + i, features.begin() + i + count, tensor->begin());
    tensors->push_back(tensor);
    if (monitor) monitor->reportProgress(i * 100 / features.size());
  }

  data->setTensors(tensors);
}

void FeaturesToTensors::writeResults() {
  if (!data)
    return;

  setTensors(data->getTensors());
}

}

}
