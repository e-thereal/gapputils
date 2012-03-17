/*
 * RbmTrainer_gpu.cu
 *
 *  Created on: Nov 10, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "RbmTrainer.h"

#include <capputils/Verifier.h>

#include <boost/progress.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>

#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

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
struct min_1 {

T  operator()(const T& x) const {
  return max((T)1, x);
}

};

template<class T>
struct minus_squared : thrust::binary_function<float, float, float> {

__host__ __device__
T operator()(const T& x, const T& y) const {
  return (x - y) * (x - y);
}

};

void RbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;
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
  const float sparsityTarget = getSparsityTarget();
  const float sparsityWeight = getSparsityWeight();

  boost::shared_ptr<RbmModel> rbm(new RbmModel());
  rbm->setIsGaussian(getIsGaussian());

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

    std::cout << "[Info] Means (" << mean << ") calculated: " << timer.elapsed() << " s" << std::endl;

    tbblas::device_vector<float>& stds = *visibleStds;
    thrust::transform(X.data().begin(), X.data().end(), X.data().begin(), _1 - mean);
    float stddev = sqrt(thrust::transform_reduce(X.data().begin(), X.data().end(),
        _1 * _1, 0.f, thrust::plus<float>()) / X.data().size());
    for (unsigned iCol = 0; iCol < X.size2(); ++iCol)
      stds(iCol) = stddev;

    std::cout << "[Info] Standard deviations calculated: " << timer.elapsed() << " s" << std::endl;

    // Apply feature scaling to training set
    thrust::transform(X.data().begin(), X.data().end(), X.data().begin(), _1 / stddev);
    std::cout << "[Info] Design matrix standardized: " << timer.elapsed() << " s" << std::endl;
  }

  for (unsigned i = X.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    tbblas::row(X, i).swap(tbblas::row(X, j));
  }
  std::cout << "[Info] Rows shuffled: " << timer.elapsed() << " s" << std::endl;

  // Train the RBM
  // Initialize weights and bias terms
  boost::shared_ptr<tbblas::device_matrix<float> > weightMatrix(new tbblas::device_matrix<float>(visibleCount, hiddenCount));
  boost::shared_ptr<tbblas::device_vector<float> > visibleBiases(new tbblas::device_vector<float>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<float> > hiddenBiases(new tbblas::device_vector<float>(hiddenCount));
  tbblas::device_matrix<float>& W = *weightMatrix;
  tbblas::device_vector<float>& b = *visibleBiases;
  tbblas::device_vector<float>& c = *hiddenBiases;

  // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)
  //std::ranlux64_base_01 eng;
  //std::normal_distribution<float> normal;

  // Initialize weights
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(W.data().size()),
    W.data().begin(), get_randn<float>(0.f, 0.1f));
  //std::for_each(W.data().begin(), W.data().end(), _1 = 1.f * normal(eng));
  //std::cout << "   E[W_ij] (c++) = " << ublas::norm_1(W) / W.data().size() << std::endl;
  //read_matrix("vishid_initial.bin", W);
  //std::cout << "E[W_ij] (matlab) = " << ublas::norm_1(W) / W.data().size() << std::endl;

  // Initialize bias terms
  thrust::fill(b.data().begin(), b.data().end(), 0.f);
  thrust::fill(c.data().begin(), c.data().end(), getInitialHidden());
  std::cout << "[Info] RBM initialized: " << timer.elapsed() << " s" << std::endl;

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
  data->setWeights(vW);

  //boost::progress_timer progresstimer;
  std::cout << "[Info] Preparation finished after " << timer.elapsed() << " s" << std::endl;
  std::cout << "[Info] Starting training" << std::endl;
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
      thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
          poshidprobs.data().begin(), sigmoid<float>());

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
      thrust::transform(
          poshidprobs.data().begin(), poshidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
          poshidstates.data().begin(), sample_units<float>()
      );

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
      thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(), neghidprobs.data().begin(),
          sigmoid<float>());

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
    std::cout << "Epoch " << iEpoch << " error " << (error / sampleCount) << " after " << timer.elapsed() << "s. ETA: "
        << hours << " h " << minutes << " min " << sec << " s" << std::endl;

    if (monitor && getShowWeights() && (iEpoch % getShowEvery() == 0)) {
      thrust::copy(W.data().begin(), W.data().begin() + vW->size(), vW->begin());
      monitor->reportProgress(100 * (iEpoch + 1) / epochCount, true);
    }
  }

  thrust::copy(W.data().begin(), W.data().begin() + vW->size(), vW->begin());
  data->setRbmModel(rbm);
  //data->setPosData(debugPosData);
  //data->setNegData(debugNegData);
}

}

}
