/*
 * RbmTrainer_gpu.cu
 *
 *  Created on: Nov 10, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "RbmTrainer.h"

#include <capputils/Logbook.h>

#include <boost/progress.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>

#include "tbblas_io.hpp"
#include "sampling.hpp"

#include <algorithm>

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

//#define TIC timer.restart();
//#define TOC cudaThreadSynchronize(); std::cout << __LINE__ << ": " << timer.elapsed() << "s" << std::endl;
//#define REPEAT for(int i = 0; i < 1000; ++i)
#define TIC
#define TOC
#define REPEAT

#define TRACE std::cout << __LINE__ << std::endl;

template<class T>
struct min_0 {
__host__ __device__
T  operator()(const T& x) const {
  return max((T)0, x);
}

};

template<class T>
struct minus_squared : thrust::binary_function<float, float, float> {

__host__ __device__
T operator()(const T& x, const T& y) const {
  return (x - y) * (x - y);
}

};

void RbmTrainer::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;
  using capputils::Severity;

  typedef tbblas::random_tensor<float, 2, true, tbblas::normal<float> > randn_t;

  boost::timer timer;
  capputils::Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  if (getVisibleCount() <= 0) {
    dlog(Severity::Warning) << "VisibleCount must be greater than 0!";
    return;
  }
  if (getTrainingSet()->size() % getVisibleCount()) {
    dlog(Severity::Warning) << "Training set size must be a multiple of VisibleCount!";
    return;
  }

  HiddenUnitType hiddenUnitType = getHiddenUnitType();

  dlog(Severity::Message) << "Building RBM with " << hiddenUnitType << " hidden units.";

  // Calculate the mean and the std of all features
  const unsigned visibleCount = getVisibleCount();
  const unsigned hiddenCount = getHiddenCount();
  const unsigned sampleCount = getTrainingSet()->size() / visibleCount;
  const float sparsityTarget = getSparsityTarget();
  const float sparsityWeight = getSparsityWeight();

  boost::shared_ptr<RbmModel> rbm(new RbmModel());
  rbm->setIsGaussian(getIsGaussian());
  rbm->setHiddenUnitType(getHiddenUnitType());

  ublas::matrix<float> trainingSet(sampleCount, visibleCount);
  std::copy(getTrainingSet()->begin(), getTrainingSet()->end(), trainingSet.data().begin());
  tbblas::device_matrix<float> X(trainingSet.size1(), trainingSet.size2());
  X = trainingSet;

  boost::shared_ptr<tbblas::device_vector<float> > visibleMeans(new tbblas::device_vector<float>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<float> > visibleStds(new tbblas::device_vector<float>(visibleCount));
  rbm->setVisibleMeans(visibleMeans);
  rbm->setVisibleStds(visibleStds);

  if (getIsGaussian()) {
    tbblas::device_vector<float>& means = *visibleMeans;
    float mean = thrust::reduce(X.data().begin(), X.data().end()) / X.data().size();

    for (unsigned iCol = 0; iCol < X.size2(); ++iCol)
      means(iCol) = mean;

    dlog() << "Means (" << mean << ") calculated: " << timer.elapsed() << " s";

    tbblas::device_vector<float>& stds = *visibleStds;
    thrust::transform(X.data().begin(), X.data().end(), X.data().begin(), _1 - mean);
    float stddev = sqrt(thrust::transform_reduce(X.data().begin(), X.data().end(),
        _1 * _1, 0.f, thrust::plus<float>()) / X.data().size());
    for (unsigned iCol = 0; iCol < X.size2(); ++iCol)
      stds(iCol) = stddev;

    dlog() << "Standard deviations calculated: " << timer.elapsed() << " s";

    // Apply feature scaling to training set
    thrust::transform(X.data().begin(), X.data().end(), X.data().begin(), _1 / stddev);
    dlog() << "Design matrix standardized: " << timer.elapsed() << " s";
  } else if (getMakeBernoulli()) {
    thrust::device_vector<float> mins(visibleCount);
    thrust::device_vector<float> maxs(visibleCount);

    for (size_t offset = 0, iVisible = 0; offset < X.data().size(); offset += sampleCount, ++iVisible) {
      assert(iVisible < visibleCount);
      float first = X.data()[offset], result;
      result = thrust::reduce(X.data().begin() + offset, X.data().begin() + offset + sampleCount,
          first, thrust::minimum<float>());
      mins[iVisible] = result;
      maxs[iVisible] = thrust::reduce(X.data().begin() + offset, X.data().begin() + offset + sampleCount,
          first, thrust::maximum<float>());
    }

    thrust::copy(mins.begin(), mins.end(), visibleMeans->data().begin());
    thrust::transform(mins.begin(), mins.end(), maxs.begin(), visibleStds->data().begin(), _2 - _1);

    for (size_t offset = 0, iVisible = 0; offset < X.data().size(); offset += sampleCount, ++iVisible) {
      assert(iVisible < visibleCount);
      thrust::transform(X.data().begin() + offset, X.data().begin() + offset + sampleCount,
          X.data().begin() + offset, _1 - (*visibleMeans)(iVisible));
      thrust::transform(X.data().begin() + offset, X.data().begin() + offset + sampleCount,
          X.data().begin() + offset, _1 / (*visibleStds)(iVisible));
    }

    ublas::matrix<float> set = X;

    boost::shared_ptr<std::vector<float> > B(new std::vector<float>(X.data().size()));
    thrust::copy(set.data().begin(), set.data().end(), B->begin());
    newState->setBernoulliData(B);
  }


  for (unsigned i = X.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    tbblas::row(X, i).swap(tbblas::row(X, j));
  }
  dlog() << "Rows shuffled: " << timer.elapsed() << " s";

  // Train the RBM
  // Initialize weights and bias terms
  boost::shared_ptr<tbblas::device_matrix<float> > weightMatrix(new tbblas::device_matrix<float>(visibleCount, hiddenCount));
  boost::shared_ptr<tbblas::device_vector<float> > visibleBiases(new tbblas::device_vector<float>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<float> > hiddenBiases(new tbblas::device_vector<float>(hiddenCount));
  tbblas::device_matrix<float>& W = *weightMatrix;
  tbblas::device_vector<float>& b = *visibleBiases;
  tbblas::device_vector<float>& c = *hiddenBiases;

  // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)

  // Initialize weights
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(W.data().size()),
    W.data().begin(), get_randn<float>(0.f, getInitialWeights()));

  // Initialize bias terms
  thrust::fill(b.data().begin(), b.data().end(), 0.f);
  thrust::fill(c.data().begin(), c.data().end(), getInitialHidden());
  dlog() << "RBM initialized: " << timer.elapsed() << " s";

  rbm->setWeightMatrix(weightMatrix);
  rbm->setVisibleBiases(visibleBiases);
  rbm->setHiddenBiases(hiddenBiases);

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

  tbblas::device_matrix<float> batch(batchSize, visibleCount);
  tbblas::device_matrix<float> negdata(batchSize, visibleCount);
  tbblas::device_matrix<float> poshidprobs(batchSize, hiddenCount);
  randn_t poshidnoise((size_t)batchSize, (size_t)hiddenCount);
  tbblas::device_matrix<float> posprods(visibleCount, hiddenCount);
  tbblas::device_matrix<float> poshidstates(batchSize, hiddenCount);
  tbblas::device_matrix<float> neghidprobs(batchSize, hiddenCount);
  tbblas::device_matrix<float> negprods(visibleCount, hiddenCount);
  tbblas::device_matrix<float> vishidinc(visibleCount, hiddenCount);
  
  tbblas::device_vector<float> hidbiasinc(hiddenCount);
  tbblas::device_vector<float> visbiasinc(visibleCount);
  tbblas::device_vector<float> poshidact(hiddenCount);
  tbblas::device_vector<float> posvisact(visibleCount);
  tbblas::device_vector<float> neghidact(hiddenCount);
  tbblas::device_vector<float> negvisact(visibleCount);

  tbblas::device_matrix<float> posdiffprobs(batchSize, hiddenCount);
  tbblas::device_vector<float> possparsityact(hiddenCount);
  tbblas::device_matrix<float> possparsityprod(visibleCount, hiddenCount);

  //boost::shared_ptr<std::vector<float> > debugPosData(new std::vector<float>(sampleCount * visibleCount));
  //boost::shared_ptr<std::vector<float> > debugNegData(new std::vector<float>(sampleCount * visibleCount));

  //std::copy(uX.data().begin(), uX.data().end(), debugPosData->begin());

  thrust::fill(vishidinc.data().begin(), vishidinc.data().end(), 0.f);
  thrust::fill(hidbiasinc.data().begin(), hidbiasinc.data().end(), 0.f);
  thrust::fill(visbiasinc.data().begin(), visbiasinc.data().end(), 0.f);

  const int epochCount = getEpochCount();

  //read_matrix("data.bin", batch);

  const int cDebugWeight = (getShowWeights() ?
      (getShowWeights() == -1 ? visibleCount * hiddenCount : getShowWeights() * visibleCount) :
      0);
  boost::shared_ptr<std::vector<float> > vW(new std::vector<float>(cDebugWeight));
  newState->setWeights(vW);

  //boost::progress_timer progresstimer;
  dlog() << "Preparation finished after " << timer.elapsed() << " s";
  dlog() << "Starting training";
  timer.restart();
  for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {
      /*** START POSITIVE PHASE ***/

      // Get current batch
      batch = tbblas::subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());

      // Calculate p(h | X, W) = sigm(XW + C)
      poshidprobs = tbblas::prod(batch, W);
      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)
        tbblas::row(poshidprobs, iRow) += c;

      switch(hiddenUnitType) {
      case HiddenUnitType::Bernoulli:
        thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
            poshidprobs.data().begin(), sigmoid<float>());
        break;
      case HiddenUnitType::ReLU:
        thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
            poshidprobs.data().begin(), min_0<float>());
        break;
      }

      // (x_n)(mu_n)'
      posprods = tbblas::prod(tbblas::trans(batch), poshidprobs);

      // Calculate the total activation of the hidden and visible units
      poshidact = tbblas::sum(poshidprobs);
      posvisact = tbblas::sum(batch);

      if (sparsityWeight != 0) {
        posdiffprobs = poshidprobs - sparsityTarget;
        possparsityact = tbblas::sum(posdiffprobs);
        possparsityprod = tbblas::prod(tbblas::trans(batch), posdiffprobs);
      }

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      if (getSampleHiddens()) {
        switch(hiddenUnitType) {
        case HiddenUnitType::Bernoulli:
          thrust::transform(
              poshidprobs.data().begin(), poshidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
              poshidstates.data().begin(), sample_units<float>()
          );
          break;

        case HiddenUnitType::ReLU:

          // Add Gaussian noise with variance sigm(x)
          //poshidprobes + sqrt(sigm(poshidprobes)) * N(0,1)
          thrust::transform(
              poshidprobs.data().begin(), poshidprobs.data().end(),
              (tbblas::sqrt(tbblas::sigm(poshidprobs)) * poshidnoise).begin(),
              poshidstates.data().begin(), _1 + _2
          );
          break;
        }
      } else {
        thrust::copy(
            poshidprobs.data().begin(), poshidprobs.data().end(), poshidstates.data().begin()
        );
      }

      /*** START NEGATIVE PHASE ***/

      // Calculate p(x | H, W) = sigm(HW' + B)
      negdata = tbblas::prod(poshidstates, tbblas::trans(W));
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(negdata, iRow) += b;

      // For the binary case
      if (!getIsGaussian()) {
        thrust::transform(negdata.begin(), negdata.end(), negdata.begin(),
            sigmoid<float>());
      }

      // Calculate p(h | Xneg, W) = sigm(XnegW + C)
      neghidprobs = tbblas::prod(negdata, W);
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(neghidprobs, iRow) += c;

      switch(hiddenUnitType) {
      case HiddenUnitType::Bernoulli:
        thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(),
            neghidprobs.data().begin(), sigmoid<float>());
        break;
      case HiddenUnitType::ReLU:
        thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(),
            neghidprobs.data().begin(), min_0<float>());
        break;
      }

      // (xneg)(mu_neg)'
      negprods = tbblas::prod(tbblas::trans(negdata), neghidprobs);

      // Calculate the total activation of the visible and hidden units (reconstruction)
      neghidact = tbblas::sum(neghidprobs);
      negvisact = tbblas::sum(negdata);

      /*** END OF NEGATIVE PHASE ***/

      if (iEpoch == epochCount - 1) {
        //ublas::matrix<float> temp = negdata;
        //thrust::copy(temp.data().begin(), temp.data().end(), debugNegData->begin() + (iBatch * batchSize * visibleCount));
      }

      float err = 0.f;
      err = tbblas::norm_2(negdata -= batch);
      error += err * err;

      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      if (sparsityWeight != 0) {
        vishidinc = momentum * vishidinc + epsilonw * ((((posprods -= negprods) += (sparsityWeight * possparsityprod)) / (float)batchSize) -= weightcost * W);
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact -= negvisact);
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * ((poshidact -= neghidact) += (sparsityWeight * possparsityact));
      } else {
        vishidinc = momentum * vishidinc + epsilonw * (((posprods -= negprods) / (float)batchSize) -= weightcost * W);
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact -= negvisact);
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact -= neghidact);
      }

      W += vishidinc;
      b += visbiasinc;
      c += hidbiasinc;

      /*** END OF UPDATES ***/

      if (monitor)
        monitor->reportProgress(100 * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    }
    int eta = timer.elapsed() / (iEpoch + 1) * (epochCount - iEpoch - 1);
    int sec = eta % 60;
    int minutes = (eta / 60) % 60;
    int hours = eta / 3600;
    dlog() << "Epoch " << iEpoch << " error " << (error / sampleCount) << " after " << timer.elapsed() << "s. ETA: "
        << hours << " h " << minutes << " min " << sec << " s";

    if (monitor && getShowWeights() && (iEpoch % getShowEvery() == 0)) {
      thrust::copy(W.data().begin(), W.data().begin() + vW->size(), vW->begin());
      monitor->reportProgress(100 * (iEpoch + 1) / epochCount, true);
    }
  }

  thrust::copy(W.data().begin(), W.data().begin() + vW->size(), vW->begin());
  newState->setRbmModel(rbm);
  //data->setPosData(debugPosData);
  //data->setNegData(debugNegData);
}

}

}
