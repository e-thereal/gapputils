/*
 * FgrbmTrainer_gpu.cu
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT

#include "FgrbmTrainer.h"

#include <algorithm>

#include <capputils/Verifier.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include "sampling.hpp"
#include "RbmModel.h"   ///< For sigmoid<T>()
#include "tbblas_io.hpp"

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

template<class T>
struct minus_squared : thrust::binary_function<T, T, T> {

T operator()(const T& x, const T& y) const {
  return (x - y) * (x - y);
}

};

template<class T>
struct add_diagonal : thrust::binary_function<unsigned, T, T> {
  unsigned diagonalShift;
  T bias;

  add_diagonal(unsigned ld, T bias) : diagonalShift(ld + 1), bias(bias) { }

  __host__ __device__
  T operator()(const unsigned& idx, const T& value) {
    return value + ((idx % diagonalShift) == 0) * bias;
  }
};

#define LOCATE(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

// TODO: This module crashs when executed, workflow reloaded and executed
//       Possibly because of cublas. Use new cublas interface in tbblas
//       and open and close a computing session accordingly
void FgrbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  boost::timer timer;

  if (!data)
    data = new FgrbmTrainer();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInitialFgrbmModel()) {
    std::cout << "[Warning] No initial model given." << std::endl;
    return;
  }

  FgrbmModel& initialModel = *getInitialFgrbmModel();

  const unsigned visibleCount = initialModel.getVisibleBiases()->size();
  const unsigned hiddenCount = initialModel.getHiddenBiases()->size();
  const unsigned factorCount = initialModel.getVisibleWeights()->size2();

  if (!getConditionalsVector() || !getVisiblesVector()) {
    std::cout << "[Warning] Missing training set!" << std::endl;
    return;
  }
  if (getVisiblesVector()->size() % visibleCount) {
    std::cout << "[Warning] Training set size must be a multiple of VisibleCount!" << std::endl;
    return;
  }
  if (getConditionalsVector()->size() != getVisiblesVector()->size()) {
    std::cout << "[Warning] Conditionals and visibles must have the same size!" << std::endl;
    return;
  }

  std::cout << "Building FGRBM ..." << std::endl;

//  FgrbmModel test;
//  LOCATE(test, VisibleMean);
//  LOCATE(test, VisibleStd);
//  LOCATE(test, VisibleBiases);
//  LOCATE(test, HiddenBiases);
//  LOCATE(test, VisibleWeights);
//  LOCATE(test, HiddenWeights);
//  LOCATE(test, ConditionalWeights);

  // Calculate the mean and the std of all features
  const unsigned sampleCount = getVisiblesVector()->size() / visibleCount;
  const int batchSize = getBatchSize();

  ublas::matrix<double> visiblesSet(sampleCount, visibleCount);
  ublas::matrix<double> conditionalsSet(sampleCount, visibleCount);
  std::copy(getVisiblesVector()->begin(), getVisiblesVector()->end(), visiblesSet.data().begin());
  std::copy(getConditionalsVector()->begin(), getConditionalsVector()->end(), conditionalsSet.data().begin());

  ublas::matrix<double>& uX = conditionalsSet;
  ublas::matrix<double>& uY = visiblesSet;

  int deviceMemory = 0;
  deviceMemory += 3 * batchSize * visibleCount;
  deviceMemory += 4 * batchSize * factorCount;
  deviceMemory += 3 * batchSize * hiddenCount;
  deviceMemory += 8 * visibleCount * factorCount;
  deviceMemory += 4 * hiddenCount * factorCount;
  deviceMemory += 4 * visibleCount;
  deviceMemory += 4 * hiddenCount;
  std::cout << "[Info] Required device memory without training data: " << 8. * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  deviceMemory += uX.size1() * uX.size2();
  deviceMemory += uY.size1() * uY.size2();
  std::cout << "[Info] Required device memory including training data: " << 8. * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  boost::shared_ptr<FgrbmModel> fgrbm = initialModel.clone();

  if (fgrbm->getIsGaussian()) {
    thrust::transform(visiblesSet.data().begin(), visiblesSet.data().end(), thrust::constant_iterator<double>(fgrbm->getVisibleMean()),
        visiblesSet.data().begin(), thrust::minus<double>());
    thrust::transform(conditionalsSet.data().begin(), conditionalsSet.data().end(), thrust::constant_iterator<double>(fgrbm->getVisibleMean()),
        conditionalsSet.data().begin(), thrust::minus<double>());

    // Apply feature scaling to training set
    thrust::transform(visiblesSet.data().begin(), visiblesSet.data().end(), thrust::constant_iterator<double>(fgrbm->getVisibleStd()),
        visiblesSet.data().begin(), thrust::divides<double>());
    thrust::transform(conditionalsSet.data().begin(), conditionalsSet.data().end(), thrust::constant_iterator<double>(fgrbm->getVisibleStd()),
        conditionalsSet.data().begin(), thrust::divides<double>());
    std::cout << "[Info] Design matrix standardized: " << timer.elapsed() << " s" << std::endl;
  }

  for (unsigned i = uX.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    ublas::row(uX, i).swap(ublas::row(uX, j));
    ublas::row(uY, i).swap(ublas::row(uY, j));
  }

  std::cout << "[Info] Rows shuffled: " << timer.elapsed() << " s" << std::endl;

  tbblas::device_matrix<double> X(uX.size1(), uX.size2());
  tbblas::device_matrix<double> Y(uY.size1(), uY.size2());
  std::cout << "[Info] Design matrices allocated: " << timer.elapsed() << " s" << std::endl;
  X = uX;
  Y = uY;

  std::cout << "[Info] Design matrices written to the device: " << timer.elapsed() << " s" << std::endl;

  // Train the RBM
  // Initialize weights and bias terms
  tbblas::device_matrix<double>& Wx = *fgrbm->getConditionalWeights();
  tbblas::device_matrix<double>& Wy = *fgrbm->getVisibleWeights();
  tbblas::device_matrix<double>& Wh = *fgrbm->getHiddenWeights();
  tbblas::device_vector<double>& b = *fgrbm->getVisibleBiases();
  tbblas::device_vector<double>& c = *fgrbm->getHiddenBiases();

  std::cout << "[Info] FGRBM initialized: " << timer.elapsed() << " s" << std::endl;

  // Start the learning
  const int batchCount = sampleCount / batchSize;
  double epsilonw =  getLearningRate();      // Learning rate for weights
  double epsilonvb = getLearningRate();      // Learning rate for biases of visible units
  double epsilonhb = getLearningRate();      // Learning rate for biases of hidden units
  double weightcost = 0; // 0.0002;
  double initialmomentum = 0.5; //65; // 0.5f;
  double finalmomentum = 0.9; // 65; // 0.9f;
  double momentum;

  tbblas::device_matrix<double> xbatch(batchSize, visibleCount);
  tbblas::device_matrix<double> ybatch(batchSize, visibleCount);
  tbblas::device_matrix<double> XWx(batchSize, factorCount);
  tbblas::device_matrix<double> YWy(batchSize, factorCount);
  tbblas::device_matrix<double> HWh(batchSize, factorCount);
  tbblas::device_matrix<double> NxF(batchSize, factorCount);
  tbblas::device_matrix<double> poshidprobs(batchSize, hiddenCount);
  tbblas::device_matrix<double> posEx(visibleCount, factorCount);
  tbblas::device_matrix<double> posEy(visibleCount, factorCount);
  tbblas::device_matrix<double> posEh(hiddenCount, factorCount);
  tbblas::device_matrix<double> poshidstates(batchSize, hiddenCount);

  tbblas::device_matrix<double> negdata(batchSize, visibleCount);
  tbblas::device_matrix<double> neghidprobs(batchSize, hiddenCount);
  tbblas::device_matrix<double> negEx(visibleCount, factorCount);
  tbblas::device_matrix<double> negEy(visibleCount, factorCount);
  tbblas::device_matrix<double> negEh(hiddenCount, factorCount);

  tbblas::device_matrix<double> Wxinc(visibleCount, factorCount);
  tbblas::device_matrix<double> Wyinc(visibleCount, factorCount);
  tbblas::device_matrix<double> Whinc(hiddenCount, factorCount);

  tbblas::device_vector<double> hidbiasinc(hiddenCount);
  tbblas::device_vector<double> visbiasinc(visibleCount);
  tbblas::device_vector<double> poshidact(hiddenCount);
  tbblas::device_vector<double> posvisact(visibleCount);
  tbblas::device_vector<double> neghidact(hiddenCount);
  tbblas::device_vector<double> negvisact(visibleCount);

  thrust::fill(Wxinc.data().begin(), Wxinc.data().end(), 0.f);
  thrust::fill(Wyinc.data().begin(), Wyinc.data().end(), 0.f);
  thrust::fill(Whinc.data().begin(), Whinc.data().end(), 0.f);
  thrust::fill(hidbiasinc.data().begin(), hidbiasinc.data().end(), 0.f);
  thrust::fill(visbiasinc.data().begin(), visbiasinc.data().end(), 0.f);

  const int epochCount = getEpochCount();

  const int wCount = visibleCount * factorCount;
  boost::shared_ptr<std::vector<float> > vWx(new std::vector<float>(wCount));
  boost::shared_ptr<std::vector<float> > vWy(new std::vector<float>(wCount));

  ublas::matrix<double, ublas::column_major> mWx = Wx;
  ublas::matrix<double, ublas::column_major> mWy = Wy;

  data->setWx(vWx);
  data->setWy(vWy);

  std::cout << "[Info] Preparation finished after " << timer.elapsed() << " s" << std::endl;
  std::cout << "[Info] Starting training" << std::endl;
  timer.restart();
  for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    double error = 0;
    for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

      /*** START POSITIVE PHASE ***/

      // Get current batch
      xbatch = tbblas::subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());
      ybatch = tbblas::subrange(Y, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());

      // Pre-compute X*Wx and Y*Wy
      XWx = tbblas::prod(xbatch, Wx);
      YWy = tbblas::prod(ybatch, Wy);

      // Calculate p(h | X, Y, W) = sigm((XWx o YWy) * WhT + C)
      poshidprobs = tbblas::prod(NxF = XWx * YWy, tbblas::trans(Wh));         // x = (XWx o YWy) * WhT
      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)             // x = x + C
        tbblas::row(poshidprobs, iRow) += c;

      thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(), // x = sigm(x)
          poshidprobs.data().begin(), sigmoid<double>());

      // Pre-compute H*Wh
      HWh = tbblas::prod(poshidprobs, Wh);

      // -dEx = XT * (YWy o HWh)
      // -dEy = YT * (XWx o HWh)
      // -dEh = HT * (XWx o YWy)
      posEx = tbblas::prod(tbblas::trans(xbatch), (NxF = YWy * HWh));
      posEy = tbblas::prod(tbblas::trans(ybatch), (NxF = XWx * HWh));
      posEh = tbblas::prod(tbblas::trans(poshidprobs), (NxF = XWx * YWy));

      // Calculate the total activation of the hidden and visible units
      poshidact = tbblas::sum(poshidprobs);
      posvisact = tbblas::sum(ybatch);

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      thrust::transform(
          poshidprobs.data().begin(), poshidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
          poshidstates.data().begin(), sample_units<double>()
      );

      /*** START NEGATIVE PHASE ***/

      // Calculate p(y | X, H, W) = sigm((X*Wx o H*Wh) * WyT + B)
      HWh = tbblas::prod(poshidstates, Wh);     // recompute using the sampled version of H
      negdata = tbblas::prod(NxF = XWx * HWh, tbblas::trans(Wy));
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(negdata, iRow) += b;

      // For the binary case
      if (!fgrbm->getIsGaussian()) {
        thrust::transform(negdata.begin(), negdata.end(), negdata.begin(),
            sigmoid<double>());

        if (getSampleVisibles()) {
          thrust::transform(
              negdata.data().begin(), negdata.data().end(), thrust::counting_iterator<unsigned>(0),
              negdata.data().begin(), sample_units<double>()
          );
        }
      } else {
        if (getSampleVisibles()) {
          thrust::transform(
              negdata.data().begin(), negdata.data().end(), thrust::counting_iterator<unsigned>(0),
              negdata.data().begin(), sample_normal<double>()
          );
        }
      }

      // Pre-compute Yneg*Wy
      YWy = tbblas::prod(negdata, Wy);

      // Calculate p(h | Yneg, X, W) = sigm((X*Wx o Yneg*Wy) * WhT + C)
      neghidprobs = tbblas::prod((NxF = XWx * YWy), tbblas::trans(Wh));         // x = (X*Wx o Yneg*Wy) * WhT
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)                   // x = x + C
        tbblas::row(neghidprobs, iRow) += c;

      thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(),   // x = sigm(x)
          neghidprobs.data().begin(), sigmoid<double>());

      // Pre-compute H*Wh
      HWh = tbblas::prod(neghidprobs, Wh);

      // -dEx = XT * (YWy o HWh)
      // -dEy = YT * (XWx o HWh)
      // -dEh = HT * (XWx o YWy)
      negEx = tbblas::prod(tbblas::trans(xbatch), (NxF = YWy * HWh));
      negEy = tbblas::prod(tbblas::trans(negdata), (NxF = XWx * HWh));
      negEh = tbblas::prod(tbblas::trans(neghidprobs), (NxF = XWx * YWy));

      // Calculate the total activation of the visible and hidden units (reconstruction)
      neghidact = tbblas::sum(neghidprobs);
      negvisact = tbblas::sum(negdata);

      /*** END OF NEGATIVE PHASE ***/

      error += tbblas::norm_1(negdata -= ybatch);
      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      if (iEpoch) {
        Wxinc = momentum * Wxinc + epsilonw * (((posEx -= negEx) / (double)batchSize) -= weightcost * Wx);
        Wyinc = momentum * Wyinc + epsilonw * (((posEy -= negEy) / (double)batchSize) -= weightcost * Wy);
        Whinc = momentum * Whinc + epsilonw * (((posEh -= negEh) / (double)batchSize) -= weightcost * Wh);
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact -= negvisact);
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact -= neghidact);
      }

      Wx += Wxinc;
      Wy += Wyinc;
      Wh += Whinc;
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

    mWx = Wx;
    mWy = Wy;
    std::copy(mWx.data().begin(), mWx.data().end(), vWx->begin());
    std::copy(mWy.data().begin(), mWy.data().end(), vWy->begin());

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / epochCount, true);
  }

  mWx = Wx;
  mWy = Wy;
  std::copy(mWx.data().begin(), mWx.data().end(), vWx->begin());
  std::copy(mWy.data().begin(), mWy.data().end(), vWy->begin());

  data->setFgrbmModel(fgrbm);
}

}

}
