/*
 * Trainer_gpu.cu
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#include "Trainer.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>
#include <tbblas/linalg.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/dot.hpp>

#include <boost/timer.hpp>

#include "math.hpp"

namespace gml {

namespace rbm {

TrainerChecker::TrainerChecker() {
  Trainer test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(HiddenCount, test);
  CHECK_MEMORY_LAYOUT2(SampleHiddens, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
  CHECK_MEMORY_LAYOUT2(InitialHidden, test);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, test);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
  CHECK_MEMORY_LAYOUT2(ShowWeights, test);
  CHECK_MEMORY_LAYOUT2(ShowEvery, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Weights, test);
}

//#define TIC timer.restart();
//#define TOC cudaThreadSynchronize(); std::cout << __LINE__ << ": " << timer.elapsed() << "s" << std::endl;
//#define REPEAT for(int i = 0; i < 1000; ++i)
#define TIC
#define TOC
#define REPEAT

#define TRACE std::cout << __LINE__ << std::endl;

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef random_tensor<value_t, 2, true, normal<value_t> > randn_t;
  typedef random_tensor<value_t, 2, true, uniform<value_t> > randu_t;

  boost::timer timer;
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  UnitType visibleUnitType = getVisibleUnitType();
  UnitType hiddenUnitType = getHiddenUnitType();

  dlog(Severity::Message) << "Building RBM with " << hiddenUnitType << " hidden units.";

  std::vector<data_t>& data = *getTrainingSet();

  // Calculate the mean and the std of all features
  const size_t visibleCount = data[0]->size();
  const size_t hiddenCount = getHiddenCount();
  const size_t sampleCount = data.size();
  const value_t sparsityTarget = getSparsityTarget();
  const value_t sparsityWeight = getSparsityWeight();

  boost::shared_ptr<Model> rbm(new Model());
  rbm->setHiddenUnitType(visibleUnitType);
  rbm->setHiddenUnitType(hiddenUnitType);

  matrix_t X(sampleCount, visibleCount);
  for (size_t i = 0; i < sampleCount; ++i) {
    thrust::copy(data[i]->begin(), data[i]->end(), row(X, i).begin());
  }

  boost::shared_ptr<host_matrix_t> visibleMeans(new host_matrix_t(zeros<value_t>(1, visibleCount)));
  boost::shared_ptr<host_matrix_t> visibleStds(new host_matrix_t(ones<value_t>(1, visibleCount)));
  rbm->setMean(visibleMeans);
  rbm->setStddev(visibleStds);

  if (visibleUnitType == UnitType::Gaussian) {
//    matrix_t means = sum(X, 0);
//    means = means / X.size()[0];
    matrix_t means = ones<value_t>(visibleMeans->size()) * sum(X) / X.count();
    X = X - repeat(means, X.size() / means.size());

//    matrix_t temp = X * X;
//    matrix_t stddev = sum(X, 0);
//    stddev = sqrt(stddev / X.size()[0]) + (stddev == 0);  // If stddev == 0 set stddev to 1
    matrix_t stddev = ones<value_t>(visibleStds->size()) * sqrt(dot(X, X) / X.count());
    X = X / repeat(stddev, X.size() / stddev.size());

    *visibleMeans = means;
    *visibleStds = stddev;
  }

  {
    matrix_t trow;
    for (unsigned i = X.size()[0] - 1; i > 0; --i) {
      unsigned j = rand() % (i + 1);
      trow = row(X, i);
      row(X, i) = row(X, j);
      row(X, j) = trow;
    }
  }
  dlog() << "Rows shuffled: " << timer.elapsed() << " s";

  // Train the RBM
  // Initialize weights and bias terms

  // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)

  matrix_t W = getInitialWeights() * randn_t(visibleCount, hiddenCount);
  matrix_t b = zeros<value_t>(1, visibleCount);
  matrix_t c = getInitialHidden() * ones<value_t>(1, hiddenCount);


  dlog() << "RBM initialized: " << timer.elapsed() << " s";

  // Start the learning
  const int batchSize = getBatchSize();
  const int batchCount = sampleCount / batchSize;
  float epsilonw =  getLearningRate();      // Learning rate for weights
  float epsilonvb = getLearningRate();      // Learning rate for biases of visible units
  float epsilonhb = getLearningRate();      // Learning rate for biases of hidden units
  float weightcost = 0.0002;
  float initialmomentum = 0.5f;
  float finalmomentum = 0.9f;
  float momentum;

  randn_t poshidnoise(batchSize, hiddenCount);
  randn_t poshidrand(batchSize, hiddenCount);
  
  matrix_t batch(batchSize, visibleCount), poshidprobs, poshidx, poshidstates, posprods,
      negdata, neghidprobs, negprods,
      posvisact, poshidact, negvisact, neghidact,
      posdiffprobs, possparsityact, possparsityprod;

  matrix_t dW = zeros<value_t>(W.size());
  matrix_t db = zeros<value_t>(b.size());
  matrix_t dc = zeros<value_t>(c.size());

  const int epochCount = getEpochCount();

  const int cDebugWeight = (getShowWeights() ?
      (getShowWeights() == -1 ? visibleCount * hiddenCount : getShowWeights() * visibleCount) :
      0);

  newState->setWeights(boost::make_shared<host_matrix_t>(W));

  //boost::progress_timer progresstimer;
  dlog() << "Preparation finished after " << timer.elapsed() << " s";
  dlog() << "Starting training";
  timer.restart();
  for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

      /*** START POSITIVE PHASE ***/

      // Get current batch
//      batch = tbblas::subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());
      batch = X[seq(iBatch * batchSize, 0), batch.size()];

      // Calculate p(h | X, W) = sigm(XW + C)
      poshidx = prod(batch, W);
      poshidx = poshidx + repeat(c, poshidx.size() / c.size());

      switch(hiddenUnitType) {
        case UnitType::Bernoulli: poshidprobs = sigm(poshidx);    break;
        case UnitType::ReLU:      poshidprobs = max(0, poshidx);  break;
        case UnitType::MyReLU:    poshidprobs = nrelu_mean(poshidx); break;
        case UnitType::ReLU1:     poshidprobs = min(1.0, max(0.0, poshidx));  break;
        case UnitType::ReLU2:     poshidprobs = min(2.0, max(0.0, poshidx));  break;
        case UnitType::ReLU4:     poshidprobs = min(4.0, max(0.0, poshidx));  break;
        case UnitType::ReLU8:     poshidprobs = min(8.0, max(0.0, poshidx));  break;
        default:
          dlog(Severity::Error) << "Hidden unit type '" << hiddenUnitType << "' has not yet been implemented.";
      }

      // (x_n)(mu_n)'
      posprods = tbblas::prod(trans(batch), poshidprobs);

      // Calculate the total activation of the hidden and visible units
      poshidact = sum(poshidprobs, 0);
      posvisact = sum(batch, 0);

      if (sparsityWeight != 0) {
        posdiffprobs = poshidprobs - sparsityTarget;
        possparsityact = sum(posdiffprobs, 0);
        possparsityprod = prod(trans(batch), posdiffprobs);
      }

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      if (getSampleHiddens()) {
        switch(hiddenUnitType) {
          case UnitType::Bernoulli: poshidstates = poshidprobs > poshidrand; break;
          case UnitType::MyReLU:
          case UnitType::ReLU:      poshidstates = max(0.0, poshidx + sqrt(sigm(poshidx)) * poshidnoise); break;
          case UnitType::ReLU1:     poshidstates = min(1.0, max(0.0, poshidx + (poshidx > 0) * (poshidx < 1.0) * poshidnoise)); break;
          case UnitType::ReLU2:     poshidstates = min(2.0, max(0.0, poshidx + (poshidx > 0) * (poshidx < 2.0) * poshidnoise)); break;
          case UnitType::ReLU4:     poshidstates = min(4.0, max(0.0, poshidx + (poshidx > 0) * (poshidx < 4.0) * poshidnoise)); break;
          case UnitType::ReLU8:     poshidstates = min(8.0, max(0.0, poshidx + (poshidx > 0) * (poshidx < 8.0) * poshidnoise)); break;
          default:
            dlog(Severity::Error) << "Hidden unit type '" << hiddenUnitType << "' has not yet been implemented.";
        }
      } else {
        poshidstates = poshidprobs;
      }

      /*** START NEGATIVE PHASE ***/

      // Calculate p(x | H, W) = sigm(HW' + B) (bernoulli case)
      negdata = prod(poshidstates, trans(W));
      negdata = negdata + repeat(b, negdata.size() / b.size());

      switch (visibleUnitType) {
        case UnitType::Bernoulli: negdata = sigm(negdata); break;
        case UnitType::Gaussian: break;
        default:
          dlog(Severity::Error) << "Visible unit type '" << visibleUnitType << "' has not yet been implemented.";
      }

      // Calculate p(h | Xneg, W) = sigm(XnegW + C)
      neghidprobs = prod(negdata, W);
      neghidprobs = neghidprobs + repeat(c, neghidprobs.size() / c.size());

      switch(hiddenUnitType) {
        case UnitType::Bernoulli: neghidprobs = sigm(neghidprobs);    break;
        case UnitType::ReLU:      neghidprobs = max(0, neghidprobs);  break;
        case UnitType::MyReLU:    neghidprobs = nrelu_mean(neghidprobs); break;
        case UnitType::ReLU1:     neghidprobs = min(1.0, max(0.0, neghidprobs));  break;
        case UnitType::ReLU2:     neghidprobs = min(2.0, max(0.0, neghidprobs));  break;
        case UnitType::ReLU4:     neghidprobs = min(4.0, max(0.0, neghidprobs));  break;
        case UnitType::ReLU8:     neghidprobs = min(8.0, max(0.0, neghidprobs));  break;
        default:
          dlog(Severity::Error) << "Hidden unit type '" << hiddenUnitType << "' has not yet been implemented.";
      }

      // (xneg)(mu_neg)'
      negprods = prod(trans(negdata), neghidprobs);

      // Calculate the total activation of the visible and hidden units (reconstruction)
      neghidact = sum(neghidprobs, 0);
      negvisact = sum(negdata, 0);

      /*** END OF NEGATIVE PHASE ***/

      error += sqrt(dot(negdata - batch, negdata - batch) / batch.count());
      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      if (sparsityWeight != 0) {
        dW = momentum * dW + epsilonw * ((posprods - negprods + sparsityWeight * possparsityprod) / batchSize - weightcost * W);
        db = momentum * db + (epsilonvb / batchSize) * (posvisact - negvisact);
        dc = momentum * dc + (epsilonhb / batchSize) * (poshidact - neghidact + sparsityWeight * possparsityact);
      } else {
        dW = momentum * dW + epsilonw * ((posprods - negprods) / batchSize - weightcost * W);
        db = momentum * db + (epsilonvb / batchSize) * (posvisact - negvisact);
        dc = momentum * dc + (epsilonhb / batchSize) * (poshidact - neghidact);
      }

      W = W + dW;
      b = b + db;
      c = c + dc;

      /*** END OF UPDATES ***/

      if (monitor)
        monitor->reportProgress(100 * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    }
    int eta = timer.elapsed() / (iEpoch + 1) * (epochCount - iEpoch - 1);
    int sec = eta % 60;
    int minutes = (eta / 60) % 60;
    int hours = eta / 3600;
    dlog() << "Epoch " << iEpoch << " error " << (error / batchCount) << " after " << timer.elapsed() << "s. ETA: "
        << hours << " h " << minutes << " min " << sec << " s";

    if (monitor && getShowWeights() && (iEpoch % getShowEvery() == 0)) {
      newState->setWeights(boost::make_shared<host_matrix_t>(W));
      monitor->reportProgress(100 * (iEpoch + 1) / epochCount, true);
    }
  }

  newState->setWeights(boost::make_shared<host_matrix_t>(W));
  rbm->setWeightMatrix(boost::make_shared<host_matrix_t>(W));
  rbm->setVisibleBiases(boost::make_shared<host_matrix_t>(b));
  rbm->setHiddenBiases(boost::make_shared<host_matrix_t>(c));
  newState->setModel(rbm);
}

}

}
