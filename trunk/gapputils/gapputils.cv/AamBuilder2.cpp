/*
 * AamBuilder2.cpp
 *
 *  Created on: Jul 20, 2011
 *      Author: tombr
 */

#include "AamBuilder2.h"

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

#include <optlib/DownhillSimplexOptimizer.h>

#include "AamEcdnll.h"

#include <algorithm>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamBuilder2)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(TrainingSet, Input("D"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InitialModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

  DefineProperty(ActiveAppearanceModel, Output("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamBuilder2::AamBuilder2() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamBuilder2");

  Changed.connect(capputils::EventHandler<AamBuilder2>(this, &AamBuilder2::changedHandler));
}

AamBuilder2::~AamBuilder2() {
  if (data)
    delete data;
}

void AamBuilder2::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamBuilder2::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamBuilder2();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int maxIter = 1;
  float variance = 1.0f, lambda = 1.0f;
  vector<float> sigma;

  const int shapeFeatureCount = getInitialModel()->getColumnCount() * getInitialModel()->getRowCount() * 2;
  const int shapeMatrixSize = shapeFeatureCount * getInitialModel()->getShapeParameterCount();

  for (int i = 0; i < maxIter; ++i) {
    AamEcdnll ecdnll(getTrainingSet(), getInitialModel(), variance, lambda, sigma);
    vector<double> parameter(shapeFeatureCount + shapeMatrixSize);
    std::copy(getInitialModel()->getMeanShape()->begin(), getInitialModel()->getMeanShape()->end(),
        parameter.begin());
    std::copy(getInitialModel()->getShapeMatrix()->begin(),
        getInitialModel()->getShapeMatrix()->end(),
        parameter.begin() + shapeFeatureCount);

    optlib::DownhillSimplexOptimizer optimizer;
    //optimizer.
    optimizer.minimize(parameter, ecdnll);

    // Minimize ecdnll -> get new shape model
    // use new shape model and old texture model to find best grids
    // Use grids and training set to update the shape and texture model
  }
}

void AamBuilder2::writeResults() {
  if (!data)
    return;

}

}

}
