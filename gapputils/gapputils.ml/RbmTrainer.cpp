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

#include <culib/lintrans.h>

#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

#include <boost/progress.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "RbmEncoder.h"
#include "RbmDecoder.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmTrainer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(TrainingSet, Input("Data"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RbmModel, Output("RBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(EpochCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(BatchSize, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(LearningRate, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmTrainer::RbmTrainer()
 : _VisibleCount(1), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01f), _IsGaussian(false), data(0)
{
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
  using namespace boost::lambda;

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
  const unsigned visibleCount = getVisibleCount();
  const unsigned hiddenCount = getHiddenCount();
  const unsigned sampleCount = getTrainingSet()->size() / visibleCount;

  boost::shared_ptr<RbmModel> rbm(new RbmModel());

  ublas::matrix<float> trainingSet(sampleCount, visibleCount);
  std::copy(getTrainingSet()->begin(), getTrainingSet()->end(), trainingSet.data().begin());

  boost::shared_ptr<ublas::vector<float> > visibleMeans(new ublas::vector<float>(visibleCount));
  boost::shared_ptr<ublas::vector<float> > visibleStds(new ublas::vector<float>(visibleCount));

  ublas::vector<float>& means = *visibleMeans;
  for (unsigned iCol = 0; iCol < trainingSet.size2(); ++iCol)
    means(iCol) = ublas::sum(ublas::column(trainingSet, iCol)) / trainingSet.size1();
  rbm->setVisibleMeans(visibleMeans);

  ublas::vector<float>& stds = *visibleStds;
  for (unsigned iCol = 0; iCol < trainingSet.size2(); ++iCol)
    stds(iCol) = sqrt(ublas::norm_2(ublas::column(trainingSet, iCol) -
      ublas::scalar_vector<float>(trainingSet.size1(), means(iCol))) / trainingSet.size1());
  rbm->setVisibleStds(visibleStds);

  // Apply feature scaling to training set
  boost::shared_ptr<ublas::matrix<float> > scaledSet = rbm->encodeDesignMatrix(trainingSet, !getIsGaussian());
  ublas::matrix<float>& X = *scaledSet;

  // Shuffle the rows of the matrix randomly
  for (unsigned i = X.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    ublas::row(X, i).swap(ublas::row(X, j));
  }

  // Train the RBM
  // Initialize weights and bias terms
  boost::shared_ptr<ublas::matrix<float> > weightMatrix(new ublas::matrix<float>(visibleCount, hiddenCount));
  boost::shared_ptr<ublas::vector<float> > visibleBiases(new ublas::vector<float>(visibleCount));
  boost::shared_ptr<ublas::vector<float> > hiddenBiases(new ublas::vector<float>(hiddenCount));
  ublas::matrix<float>& W = *weightMatrix;
  ublas::vector<float>& b = *visibleBiases;
  ublas::vector<float>& c = *hiddenBiases;

  // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)
  std::ranlux64_base_01 eng;
  std::normal_distribution<float> normal;

  // Initialize weights
  std::for_each(W.data().begin(), W.data().end(), _1 = 0.1f * normal(eng));

  // Initialize bias terms
  std::fill(b.begin(), b.end(), 0.f);
  std::fill(c.begin(), c.end(), 0.f);

  rbm->setWeightMatrix(weightMatrix);
  rbm->setVisibleBiases(visibleBiases);
  rbm->setHiddenBiases(hiddenBiases);

  // Start the learning
  const int batchSize = getBatchSize();
  const int batchCount = sampleCount / batchSize;
  float epsilonw =  getLearningRate();      // Learning rate for weights
  float epsilonvb = getLearningRate();      // Learning rate for biases of visible units
  float epsilonhb = getLearningRate();      // Learning rate for biases of hidden units
  float weightcost = 1e-2 * getLearningRate();
  float initialmomentum = 0.5f;
  float finalmomentum = 0.9f;
  float momentum;

  ublas::matrix<float> batch(batchSize, visibleCount);
  ublas::matrix<float> negdata(batchSize, visibleCount);
  ublas::matrix<float> diffdata(batchSize, visibleCount);
  ublas::matrix<float> poshidprobs(batchSize, hiddenCount);
  ublas::matrix<float> posprods(visibleCount, hiddenCount);
  ublas::matrix<float> poshidstates(batchSize, hiddenCount);
  ublas::matrix<float> neghidprobs(batchSize, hiddenCount);
  ublas::matrix<float> negprods(visibleCount, hiddenCount);
  ublas::matrix<float> vishidinc(visibleCount, hiddenCount);
  ublas::vector<float> hidbiasinc(hiddenCount);
  ublas::vector<float> visbiasinc(visibleCount);
  ublas::vector<float> poshidact(hiddenCount);
  ublas::vector<float> posvisact(visibleCount);
  ublas::vector<float> neghidact(hiddenCount);
  ublas::vector<float> negvisact(visibleCount);

  std::fill(vishidinc.data().begin(), vishidinc.data().end(), 0.f);
  std::fill(hidbiasinc.begin(), hidbiasinc.end(), 0.f);
  std::fill(visbiasinc.begin(), visbiasinc.end(), 0.f);

  const int epochCount = getEpochCount();

  boost::progress_timer timer;
  for (int iEpoch = 0; iEpoch < epochCount; ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      /*** START POSITIVE PHASE ***/

      // Get current batch
      batch = subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());

      // Calculate p(h | X, W) = sigm(XW + C)
      poshidprobs = prod(batch, W);
      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)
        ublas::row(poshidprobs, iRow) += c;

      std::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
          poshidprobs.data().begin(), sigmoid);

      // (x_n)(mu_n)'
      posprods = ublas::prod(ublas::trans(batch), poshidprobs);

      // Calculate the total activation of the hidden and visible units
      for (unsigned iCol = 0; iCol < poshidprobs.size2(); ++iCol)
        poshidact(iCol) = ublas::sum(ublas::column(poshidprobs, iCol));
      for (unsigned iCol = 0; iCol < batch.size2(); ++iCol)
        posvisact(iCol) = ublas::sum(ublas::column(batch, iCol));

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      std::transform(poshidprobs.data().begin(), poshidprobs.data().end(), poshidstates.data().begin(),
          _1 > ((float)rand() / (float)RAND_MAX));

      /*** START NEGATIVE PHASE ***/

      // Calculate p(x | H, W) = sigm(HW' + B)
      negdata = prod(poshidstates, trans(W));
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        ublas::row(negdata, iRow) += b;

      // For the binary case
      if (!getIsGaussian())
        std::transform(negdata.data().begin(), negdata.data().end(), negdata.data().begin(),
            sigmoid);

      // Calculate p(h | Xneg, W) = sigm(XnegW + C)
      neghidprobs = prod(negdata, W);
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        ublas::row(neghidprobs, iRow) += c;

      std::transform(neghidprobs.data().begin(), neghidprobs.data().end(), neghidprobs.data().begin(),
          sigmoid);

      // (xneg)(mu_neg)'
      negprods = prod(trans(negdata), neghidprobs);

      // Calculate the total activation of the visible and hidden units (reconstruction)
      for (unsigned iCol = 0; iCol < neghidprobs.size2(); ++iCol)
        neghidact(iCol) = ublas::sum(ublas::column(neghidprobs, iCol));
      for (unsigned iCol = 0; iCol < negdata.size2(); ++iCol)
        negvisact(iCol) = ublas::sum(ublas::column(negdata, iCol));

      /*** END OF NEGATIVE PHASE ***/

      diffdata = batch - negdata;
      for (unsigned iRow = 0; iRow < diffdata.size1(); ++iRow)
        error += ublas::norm_2(ublas::row(diffdata, iRow));

      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      if (iEpoch) {
        // Don't learn anything in the first epoch in order to get a good estimate of the initial error
        vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / batchSize - weightcost * W);
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact - negvisact);
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact - neghidact);
      }

      W += vishidinc;
      b += visbiasinc;
      c += hidbiasinc;

      /*** END OF UPDATES ***/

      if (monitor)
        monitor->reportProgress(100 * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    }
    std::cout << "Epoch " << iEpoch << " error " << error << std::endl;
  }

//
//  // Batched implementation
////  int maxEpoch = 0;
////  int batchSize = 10;
////  int batchCount = sampleCount / batchSize;
////  for (int iEpoch = 0; iEpoch < maxEpoch; ++iEpoch) {
////    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {
////      // set gradient to zero (we have a gradient for each bias term and one for the weight matrix
////      for (int iSample = 0; iSample < batchSize; ++iSample) {
////        const int currentSample = iBatch * batchSize + iSample;
////        // Compute mu_currentSample = E[h | x_currentSample, W]
////
////        // Sample h_currentSample ~ p(h | x_currentSample, W)
////        // Sample x'_currentSample ~ p(x | h_currentSample, W)
////        if (getSampleHiddens()) {
////          // sample h'_currentSample ~ p(h |x'_currentSample, W)
////          // g += (x_currentSample)(mu_currentSample)T - (x'_currentSample)(h'_currentSample)T
////        } else {
////          // Compute mu'_currentSample = E[h | x'_currentSample, W]
////          // g += (x_currentSample)(mu_currentSample)T - (x'_currentSample)(mu'_currentSample)T
////        }
////      }
////      // Update parameters W += (alpha_iEpoch/batchSize)g
////    }
////  }
//
//  // Unbatched implementation
//  int maxEpoch = 10;
//  float alpha = 1e-4;
//  for (int iEpoch = 0; iEpoch < maxEpoch; ++iEpoch) {
//    // Compute mu = E[h | x, W]
//    boost::shared_ptr<std::vector<float> > mu = RbmEncoder::getExpectations(scaledSet.get(), rbm.get());
//
//    // Sample h ~ p(h | x, W)
//    boost::shared_ptr<std::vector<float> > h = RbmEncoder::sampleHiddens(mu.get());
//
//    // Sample x' ~ p(x | h, W)
//    // TODO: visibles are gaussian
//    boost::shared_ptr<std::vector<float> > xPrime = RbmDecoder::sampleVisibles(h.get(), rbm.get(), true);
//
//    // Compute mu' = E[h | x', W]
//    boost::shared_ptr<std::vector<float> > muPrime = RbmEncoder::getExpectations(h.get(), rbm.get());
//
//    if (getSampleHiddens()) {
//      // sample h' ~ p(h |x', W)
//      // g = (x)(mu)T - (x')(h')T
//      // Update parameters W += (alpha_iEpoch)g
//    } else {
//      // g = (x)(mu)T - (x')(mu')T
//      matrix<float> _x(sampleCount, visibleCount + 1);
//      matrix<float> _mu(sampleCount, hiddenCount + 1);
//      matrix<float> _xPrime(sampleCount, visibleCount + 1);
//      matrix<float> _muPrime(sampleCount, hiddenCount + 1);
//      matrix<float> g(visibleCount + 1, hiddenCount + 1);
//
//      std::copy(scaledSet->begin(), scaledSet->end(), _x.data().begin());
//      std::copy(mu->begin(), mu->end(), _mu.data().begin());
//      std::copy(xPrime->begin(), xPrime->end(), _xPrime.data().begin());
//      std::copy(muPrime->begin(), muPrime->end(), _muPrime.data().begin());
//
//      g = prod(trans(_x), _mu) - prod(trans(_xPrime), _muPrime);
//      // Update parameters W += (alpha_iEpoch)g
//      for (unsigned i = 0; i < (visibleCount + 1) * (hiddenCount + 1); ++i)
//        weightMatrix->at(i) += alpha * g.data()[i];
//    }
//
//    if (monitor)
//      monitor->reportProgress(100 * (iEpoch + 1) / maxEpoch);
//  }

  data->setRbmModel(rbm);
}

void RbmTrainer::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
}

}

}
