/*
 * RbmTrainer.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmTrainer.h"

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

#include <cmath>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmTrainer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(TrainingSet, Input("Data"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RbmModel, Output("RBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmTrainer::RbmTrainer() : _VisibleCount(1), _HiddenCount(1), _SampleHiddens(true), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmTrainer");

  Changed.connect(capputils::EventHandler<RbmTrainer>(this, &RbmTrainer::changedHandler));
}

RbmTrainer::~RbmTrainer() {
  if (data)
    delete data;
}

void RbmTrainer::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

template <class T>
T square(const T& a) { return a * a; }

void RbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmTrainer();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getTrainingSet()) {
    std::cout << "[Warning] Missing training set!" << std::endl;
    return;
  } else if (getVisibleCount() <= 0) {
    std::cout << "[Warning] VisibleCount must be greater than 0!" << std::endl;
    return;
  } else if (getTrainingSet()->size() % getVisibleCount()) {
    std::cout << "[Warning] Training set size must be a multiple of VisibleCount!" << std::endl;
    return;
  }

  std::cout << "Building RBM ..." << std::endl;

  // Calculate the mean and the std of all features
  std::vector<float>* trainingSet = getTrainingSet().get();
  const unsigned visibleCount = getVisibleCount();
  const unsigned hiddenCount = getHiddenCount();
  const unsigned sampleCount = trainingSet->size() / visibleCount;

  boost::shared_ptr<RbmModel> rbm(new RbmModel());
  rbm->setVisibleCount(visibleCount);
  rbm->setHiddenCount(hiddenCount);

  // Calculate the means of the visible units
  boost::shared_ptr<std::vector<float> > visibleMeans(new std::vector<float>(visibleCount, 0.f));
  for (unsigned i = 0; i < trainingSet->size(); ++i)
    (*visibleMeans)[i % visibleCount] += trainingSet->at(i);
  for (unsigned i = 0; i < visibleCount; ++i)
    (*visibleMeans)[i] /= (float)sampleCount;
  rbm->setVisibleMeans(visibleMeans);

  // Calculate the standard deviation of the visible units
  boost::shared_ptr<std::vector<float> > visibleStds(new std::vector<float>(visibleCount, 0.f));
  for (unsigned i = 0; i < trainingSet->size(); ++i)
    (*visibleStds)[i % visibleCount] += square(trainingSet->at(i) - visibleMeans->at(i % visibleCount));
  for (unsigned i = 0; i < visibleCount; ++i)
    (*visibleStds)[i] = sqrt(visibleStds->at(i) / (float)sampleCount);
  rbm->setVisibleStds(visibleStds);

  // Apply feature scaling to training set and add a first column of 1's
  boost::shared_ptr<std::vector<float> > scaledSet = rbm->encodeDesignMatrix(trainingSet);

  // Train the RBM
  // Initialize weights and bias terms
  boost::shared_ptr<std::vector<float> > weightMatrix(new std::vector<float>((visibleCount+1) * (hiddenCount+1), 0.f));
  // make weightMatrix diagonal for testing purpose
  for (unsigned i = 0, j = 0; i < weightMatrix->size() && j < hiddenCount; i += hiddenCount + 2, ++j)
    weightMatrix->at(i) = 1.f;
  int maxEpoch = 0;
  int batchSize = 10;
  int batchCount = sampleCount / batchSize;
  for (int iEpoch = 0; iEpoch < maxEpoch; ++iEpoch) {
    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {
      // set gradient to zero (we have a gradient for each bias term and one for the weight matrix
      for (int iSample = 0; iSample < batchSize; ++iSample) {
        const int currentSample = iBatch * batchSize + iSample;
        // Compute mu_currentSample = E[h | x_currentSample, W]
        // Sample h_currentSample ~ p(h | x_currentSample, W)
        // Sample x'_currentSample ~ p(x | h_currentSample, W)
        if (getSampleHiddens()) {
          // sample h'_currentSample ~ p(h |x'_currentSample, W)
          // g += (x_currentSample)(mu_currentSample)T - (x'_currentSample)(h'_currentSample)T
        } else {
          // Compute mu'_currentSample = E[h | x'_currentSample, W]
          // g += (x_currentSample)(mu_currentSample)T - (x'_currentSample)(mu'_currentSample)T
        }
      }
      // Update parameters W += (alpha_iEpoch/batchSize)g
    }
  }
  rbm->setWeightMatrix(weightMatrix);

  data->setRbmModel(rbm);
}

void RbmTrainer::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
}

}

}
