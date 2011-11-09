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
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include "RbmEncoder.h"
#include "RbmDecoder.h"
#include "ublas_io.hpp"

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
  boost::timer timer;

  if (!data)
    data = new RbmTrainer();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getTrainingSet()) {
    std::cout << "[Warning] Missing training set!" << std::endl;
    return;
  }
  if (getVisibleCount() <= 0) {
    std::cout << "[Warning] VisibleCount must be greater than 0!" << std::endl;
    return;
  } 
  if (getTrainingSet()->size() % getVisibleCount()) {
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
  boost::shared_ptr<ublas::matrix<float, ublas::column_major> > weightMatrix(new ublas::matrix<float, ublas::column_major>(visibleCount, hiddenCount));
  boost::shared_ptr<ublas::vector<float> > visibleBiases(new ublas::vector<float>(visibleCount));
  boost::shared_ptr<ublas::vector<float> > hiddenBiases(new ublas::vector<float>(hiddenCount));
  ublas::matrix<float, ublas::column_major>& W = *weightMatrix;
  ublas::vector<float>& b = *visibleBiases;
  ublas::vector<float>& c = *hiddenBiases;

  // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)
  std::ranlux64_base_01 eng;
  std::normal_distribution<float> normal;

  // Initialize weights
  std::for_each(W.data().begin(), W.data().end(), _1 = 1.f * normal(eng));
  std::cout << "   E[W_ij] (c++) = " << ublas::norm_1(W) / W.data().size() << std::endl;
  read_matrix("vishid_initial.bin", W);
  std::cout << "E[W_ij] (matlab) = " << ublas::norm_1(W) / W.data().size() << std::endl;

  // Initialize bias terms
  std::fill(b.begin(), b.end(), 0.f);
  std::fill(c.begin(), c.end(), 0.f);

  //rbm->setWeightMatrix(weightMatrix);
  rbm->setVisibleBiases(visibleBiases);
  rbm->setHiddenBiases(hiddenBiases);

  // Start the learning
  const int batchSize = getBatchSize();
  const int batchCount = sampleCount / batchSize;
  float epsilonw =  0.1;      // Learning rate for weights
  float epsilonvb = 0.1;      // Learning rate for biases of visible units
  float epsilonhb = 0.1;      // Learning rate for biases of hidden units
  float weightcost = 0.0002;
  float initialmomentum = 0.5f;
  float finalmomentum = 0.9f;
  float momentum;

  ublas::matrix<float, ublas::column_major> batch(batchSize, visibleCount);
  ublas::matrix<float> negdata(batchSize, visibleCount);
  ublas::matrix<float> diffdata(batchSize, visibleCount);
  ublas::matrix<float> poshidprobs(batchSize, hiddenCount);
  ublas::matrix<float> posprods(visibleCount, hiddenCount);
  ublas::matrix<float, ublas::column_major> poshidstates(batchSize, hiddenCount);
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

  boost::progress_timer progresstimer;
  std::cout << "[Info] Start calculation" << std::endl;
  for (int iEpoch = 0; iEpoch < epochCount; ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      /*** START POSITIVE PHASE ***/

      // Get current batch
      batch = subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());
      read_matrix("data.bin", batch);

      // Calculate p(h | X, W) = sigm(XW + C)
      timer.restart();
      poshidprobs = prod(batch, W);
      std::cout << "poshidprobs (prod): " << timer.elapsed() << std::endl;
      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)
        ublas::row(poshidprobs, iRow) += c;
      std::cout << "poshidprobs (-C): " << timer.elapsed() << std::endl;

      std::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
          poshidprobs.data().begin(), sigmoid);
      std::cout << "poshidprobs (sigmoid): " << timer.elapsed() << std::endl;

      // (x_n)(mu_n)'
      timer.restart();
      posprods = ublas::prod(ublas::trans(batch), poshidprobs);
      std::cout << "posprods: " << timer.elapsed() << std::endl;

      // Calculate the total activation of the hidden and visible units
      timer.restart();
      for (unsigned iCol = 0; iCol < poshidprobs.size2(); ++iCol)
        poshidact(iCol) = ublas::sum(ublas::column(poshidprobs, iCol));
      std::cout << "poshidact: " << timer.elapsed() << std::endl;

      timer.restart();
      for (unsigned iCol = 0; iCol < batch.size2(); ++iCol)
        posvisact(iCol) = ublas::sum(ublas::column(batch, iCol));
      std::cout << "posvisact: " << timer.elapsed() << std::endl;

      ublas::matrix<float, ublas::column_major> posprods_test;
      ublas::vector<float> poshidact_test, posvisact_test;
      read_matrix("posprods.bin", posprods_test);
      read_vector("poshidact.bin", poshidact_test);
      read_vector("posvisact.bin", posvisact_test);

      std::cout << "Positive phase:" << std::endl;
      std::cout << ublas::norm_1(posprods - posprods_test) / posprods.data().size() << std::endl;
      std::cout << ublas::norm_1(poshidact - poshidact_test) / poshidact.size() << std::endl;
      std::cout << ublas::norm_1(posvisact - posvisact_test) / posvisact.size() << std:: endl;

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      timer.restart();
      std::transform(poshidprobs.data().begin(), poshidprobs.data().end(), poshidstates.data().begin(),
          _1 > ((float)rand() / (float)RAND_MAX));
      std::cout << "poshidstates: " << timer.elapsed() << std::endl;

      std::cout << "      E[p(h)] = " << ublas::norm_1(poshidprobs) / poshidprobs.data().size() << std::endl;
      std::cout << " E[h] (ublas) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;
      
      read_matrix("poshidstates.bin", poshidstates);
      std::cout << "E[h] (matlab) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;

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

      ublas::matrix<float, ublas::column_major> negprods_test;
      ublas::vector<float> neghidact_test, negvisact_test;
      read_matrix("negprods.bin", negprods_test);
      read_vector("neghidact.bin", neghidact_test);
      read_vector("negvisact.bin", negvisact_test);

      std::cout << "Negative phase:" << std::endl;
      std::cout << ublas::norm_1(negprods - negprods_test) / negprods.data().size() << std::endl;
      std::cout << ublas::norm_1(neghidact - neghidact_test) / neghidact.size() << std::endl;
      std::cout << ublas::norm_1(negvisact - negvisact_test) / negvisact.size() << std:: endl;

      /*** END OF NEGATIVE PHASE ***/

      diffdata = batch - negdata;
      for (unsigned iRow = 0; iRow < diffdata.size1(); ++iRow)
        error += ublas::norm_2(ublas::row(diffdata, iRow));

      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      //if (iEpoch) {
        // Don't learn anything in the first epoch in order to get a good estimate of the initial error
        vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / batchSize - weightcost * W);
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact - negvisact);
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact - neghidact);
      //}

      ublas::matrix<float, ublas::column_major> vishidinc_test;
      ublas::vector<float> visbiasinc_test, hidbiasinc_test;
      read_matrix("vishidinc.bin", vishidinc_test);
      read_vector("visbiasinc.bin", visbiasinc_test);
      read_vector("hidbiasinc.bin", hidbiasinc_test);

      std::cout << "Finalization phase:" << std::endl;
      std::cout << ublas::norm_1(vishidinc - vishidinc_test) / vishidinc.data().size() << std::endl;
      std::cout << ublas::norm_1(visbiasinc - visbiasinc_test) / visbiasinc.size() << std::endl;
      std::cout << ublas::norm_1(hidbiasinc - hidbiasinc_test) / hidbiasinc.size() << std:: endl;

      W += vishidinc;
      b += visbiasinc;
      c += hidbiasinc;

      /*** END OF UPDATES ***/

      if (monitor)
        monitor->reportProgress(100 * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    }
    std::cout << "Epoch " << iEpoch << " error " << error << std::endl;
  }

  data->setRbmModel(rbm);
}

void RbmTrainer::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
}

}

}
