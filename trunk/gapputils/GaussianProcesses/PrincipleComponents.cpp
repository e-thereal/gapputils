/*
 * PrincipleComponents.cpp
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#include "PrincipleComponents.h"

#include <capputils/DescriptionAttribute.h>
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

#include <culapackdevice.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(PrincipleComponents)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(FeatureCount, Input("D"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleCount, Input("N"), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Data, Input(), Description("All features must be normalized to have zero mean"), Volatile(), Hide(), NotEqual<double*>(0), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(PrincipleComponents, Output("PC"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

PrincipleComponents::PrincipleComponents() : _FeatureCount(0), _SampleCount(0), _Data(0), _PrincipleComponents(0), data(0) {
  WfeUpdateTimestamp
  setLabel("PrincipleComponents");

  Changed.connect(capputils::EventHandler<PrincipleComponents>(this, &PrincipleComponents::changedHandler));
}

PrincipleComponents::~PrincipleComponents() {
  if (data)
    delete data;

  if (_PrincipleComponents)
    delete _PrincipleComponents;
}

void PrincipleComponents::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void PrincipleComponents::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new PrincipleComponents();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int m = getFeatureCount();  // number of rows (column major order is assumed)
  const int n = getSampleCount();   // number of cols (column major order is assumed)

  double* pcs = new double[m * m];

  getPcs(pcs, getData(), m, n);

  if (data->getPrincipleComponents())
    delete data->getPrincipleComponents();
  data->setPrincipleComponents(pcs);
}

void PrincipleComponents::writeResults() {
  if (!data)
    return;

  if (getPrincipleComponents())
    delete getPrincipleComponents();
  setPrincipleComponents(data->getPrincipleComponents());
  data->setPrincipleComponents(0);
}

}
