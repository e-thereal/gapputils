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

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(PrincipleComponents)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(FeatureCount, Input("D"), Observe(Id), TimeStamp(Id))
  DefineProperty(Data, Input(), Description("All features must be normalized to have zero mean"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(PrincipleComponents, Output("PC"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

PrincipleComponents::PrincipleComponents() : _FeatureCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("PrincipleComponents");

  Changed.connect(capputils::EventHandler<PrincipleComponents>(this, &PrincipleComponents::changedHandler));
}

PrincipleComponents::~PrincipleComponents() {
  if (data)
    delete data;
}

void PrincipleComponents::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void PrincipleComponents::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new PrincipleComponents();

  if (!capputils::Verifier::Valid(*this))
    return;

  std::vector<float>& input = *getData();

  const int m = getFeatureCount();  // number of rows (column major order is assumed)
  const int n = input.size() / m;

  boost::shared_ptr<std::vector<float> > pcs(new std::vector<float>(m * m));

  getPcs(&pcs->at(0), &input[0], m, n);

  data->setPrincipleComponents(pcs);
}

void PrincipleComponents::writeResults() {
  if (!data)
    return;

  setPrincipleComponents(data->getPrincipleComponents());
}

}

}
