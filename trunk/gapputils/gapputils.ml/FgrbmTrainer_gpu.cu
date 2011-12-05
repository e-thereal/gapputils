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

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

template<class T>
struct minus_squared : thrust::binary_function<float, float, float> {

T operator()(const T& x, const T& y) const {
  return (x - y) * (x - y);
}

};

// TODO: This module crashs when executed, workflow reloaded and executed
//       Possibly because of cublas. Use new cublas interface in tbblas
//       and open and close a computing session accordingly
void FgrbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  boost::timer timer;

  if (!data)
    data = new FgrbmTrainer();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getConditionalsVector() || !getVisiblesVector()) {
    std::cout << "[Warning] Missing training set!" << std::endl;
    return;
  }
  if (getVisibleCount() <= 0) {
    std::cout << "[Warning] VisibleCount must be greater than 0!" << std::endl;
    return;
  }
  if (getVisiblesVector()->size() % getVisibleCount()) {
    std::cout << "[Warning] Training set size must be a multiple of VisibleCount!" << std::endl;
    return;
  }
  if (getConditionalsVector()->size() != getVisiblesVector()->size()) {
    std::cout << "[Warning] Conditionals and visibles must have the same size!" << std::endl;
    return;
  }

  std::cout << "Building FGRBM ..." << std::endl;

  // Calculate the mean and the std of all features
  const unsigned visibleCount = getVisibleCount();
  const unsigned hiddenCount = getHiddenCount();
  const unsigned factorCount = getFactorCount();
  const unsigned sampleCount = getVisiblesVector()->size() / visibleCount;
  const int batchSize = getBatchSize();

  boost::shared_ptr<FgrbmModel> fgrbm(new FgrbmModel());

  ublas::matrix<float> visiblesSet(sampleCount, visibleCount);
  ublas::matrix<float> conditionalsSet(sampleCount, visibleCount);
  std::copy(getVisiblesVector()->begin(), getVisiblesVector()->end(), visiblesSet.data().begin());
  std::copy(getConditionalsVector()->begin(), getConditionalsVector()->end(), conditionalsSet.data().begin());

  float mean = thrust::reduce(visiblesSet.data().begin(), visiblesSet.data().end()) / visiblesSet.data().size();
  fgrbm->setVisibleMean(mean);
  std::cout << "[Info] Means calculated: " << timer.elapsed() << " s (" << mean << ")" << std::endl;

  if (getIsGaussian()) {
    thrust::transform(visiblesSet.data().begin(), visiblesSet.data().end(), thrust::constant_iterator<float>(mean),
        visiblesSet.data().begin(), thrust::minus<float>());
    thrust::transform(conditionalsSet.data().begin(), conditionalsSet.data().end(), thrust::constant_iterator<float>(mean),
        conditionalsSet.data().begin(), thrust::minus<float>());

    float stddev = sqrt(thrust::inner_product(visiblesSet.data().begin(), visiblesSet.data().end(),
        visiblesSet.data().begin(), 0.f) / visiblesSet.data().size());
    fgrbm->setVisibleStd(stddev);
    std::cout << "[Info] Standard deviations calculated: " << timer.elapsed() << " s" << std::endl;

    // Apply feature scaling to training set
    thrust::transform(visiblesSet.data().begin(), visiblesSet.data().end(), thrust::constant_iterator<float>(stddev),
        visiblesSet.data().begin(), thrust::divides<float>());
    thrust::transform(conditionalsSet.data().begin(), conditionalsSet.data().end(), thrust::constant_iterator<float>(stddev),
        conditionalsSet.data().begin(), thrust::divides<float>());
    std::cout << "[Info] Design matrix standardized: " << timer.elapsed() << " s" << std::endl;
  }
  ublas::matrix<float>& uX = conditionalsSet;
  ublas::matrix<float>& uY = visiblesSet;

  for (unsigned i = uX.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    ublas::row(uX, i).swap(ublas::row(uX, j));
    ublas::row(uY, i).swap(ublas::row(uY, j));
  }
  std::cout << "[Info] Rows shuffled: " << timer.elapsed() << " s" << std::endl;

  int deviceMemory = 0;
  deviceMemory += 3 * batchSize * visibleCount;
  deviceMemory += 4 * batchSize * factorCount;
  deviceMemory += 3 * batchSize * hiddenCount;
  deviceMemory += 8 * visibleCount * factorCount;
  deviceMemory += 4 * hiddenCount * factorCount;
  deviceMemory += 4 * visibleCount;
  deviceMemory += 4 * hiddenCount;
  std::cout << "[Info] Required device memory without training data: " << 4. * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  deviceMemory += uX.size1() * uX.size2();
  deviceMemory += uY.size1() * uY.size2();
  std::cout << "[Info] Required device memory including training data: " << 4. * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  tbblas::device_matrix<float> X(uX.size1(), uX.size2());
  tbblas::device_matrix<float> Y(uY.size1(), uY.size2());
  std::cout << "[Info] Design matrices allocated: " << timer.elapsed() << " s" << std::endl;
  X = uX;
  Y = uY;
  std::cout << "[Info] Design matrices written to the device: " << timer.elapsed() << " s" << std::endl;

  // Train the RBM
  // Initialize weights and bias terms
  boost::shared_ptr<tbblas::device_matrix<float> > conditionalWeights(new tbblas::device_matrix<float>(visibleCount, factorCount));
  boost::shared_ptr<tbblas::device_matrix<float> > visibleWeights(new tbblas::device_matrix<float>(visibleCount, factorCount));
  boost::shared_ptr<tbblas::device_matrix<float> > hiddenWeights(new tbblas::device_matrix<float>(hiddenCount, factorCount));
  boost::shared_ptr<tbblas::device_vector<float> > visibleBiases(new tbblas::device_vector<float>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<float> > hiddenBiases(new tbblas::device_vector<float>(hiddenCount));
  tbblas::device_matrix<float>& Wx = *conditionalWeights;
  tbblas::device_matrix<float>& Wy = *visibleWeights;
  tbblas::device_matrix<float>& Wh = *hiddenWeights;
  tbblas::device_vector<float>& b = *visibleBiases;
  tbblas::device_vector<float>& c = *hiddenBiases;

  // Initialize weights
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wx.data().size()),
      Wx.data().begin(), get_randn<float>(0.f, 0.01f));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wy.data().size()),
      Wy.data().begin(), get_randn<float>(0.f, 0.01f));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wh.data().size()),
      Wh.data().begin(), get_randn<float>(0.f, 0.01f));

  // Initialize bias terms
  thrust::fill(b.data().begin(), b.data().end(), 0.f);
  thrust::fill(c.data().begin(), c.data().end(), getInitialHidden());
  std::cout << "[Info] FGRBM initialized: " << timer.elapsed() << " s" << std::endl;

  fgrbm->setConditionalWeights(conditionalWeights);
  fgrbm->setVisibleWeights(visibleWeights);
  fgrbm->setHiddenWeights(hiddenWeights);
  fgrbm->setVisibleBiases(visibleBiases);
  fgrbm->setHiddenBiases(hiddenBiases);

  // Start the learning
  const int batchCount = sampleCount / batchSize;
  float epsilonw =  getLearningRate();      // Learning rate for weights
  float epsilonvb = getLearningRate();      // Learning rate for biases of visible units
  float epsilonhb = getLearningRate();      // Learning rate for biases of hidden units
  float weightcost = 0.f; // 0.0002;
  float initialmomentum = 0.5f;
  float finalmomentum = 0.9f;
  float momentum;

  tbblas::device_matrix<float> xbatch(batchSize, visibleCount);
  tbblas::device_matrix<float> ybatch(batchSize, visibleCount);
  tbblas::device_matrix<float> XWx(batchSize, factorCount);
  tbblas::device_matrix<float> YWy(batchSize, factorCount);
  tbblas::device_matrix<float> HWh(batchSize, factorCount);
  tbblas::device_matrix<float> NxF(batchSize, factorCount);
  tbblas::device_matrix<float> poshidprobs(batchSize, hiddenCount);
  tbblas::device_matrix<float> posEx(visibleCount, factorCount);
  tbblas::device_matrix<float> posEy(visibleCount, factorCount);
  tbblas::device_matrix<float> posEh(hiddenCount, factorCount);
  tbblas::device_matrix<float> poshidstates(batchSize, hiddenCount);

  tbblas::device_matrix<float> negdata(batchSize, visibleCount);
  tbblas::device_matrix<float> neghidprobs(batchSize, hiddenCount);
  tbblas::device_matrix<float> negEx(visibleCount, factorCount);
  tbblas::device_matrix<float> negEy(visibleCount, factorCount);
  tbblas::device_matrix<float> negEh(hiddenCount, factorCount);

  tbblas::device_matrix<float> Wxinc(visibleCount, factorCount);
  tbblas::device_matrix<float> Wyinc(visibleCount, factorCount);
  tbblas::device_matrix<float> Whinc(hiddenCount, factorCount);

  tbblas::device_vector<float> hidbiasinc(hiddenCount);
  tbblas::device_vector<float> visbiasinc(visibleCount);
  tbblas::device_vector<float> poshidact(hiddenCount);
  tbblas::device_vector<float> posvisact(visibleCount);
  tbblas::device_vector<float> neghidact(hiddenCount);
  tbblas::device_vector<float> negvisact(visibleCount);

  thrust::fill(Wxinc.data().begin(), Wxinc.data().end(), 0.f);
  thrust::fill(Wyinc.data().begin(), Wyinc.data().end(), 0.f);
  thrust::fill(Whinc.data().begin(), Whinc.data().end(), 0.f);
  thrust::fill(hidbiasinc.data().begin(), hidbiasinc.data().end(), 0.f);
  thrust::fill(visbiasinc.data().begin(), visbiasinc.data().end(), 0.f);

  const int epochCount = getEpochCount();

  std::cout << "[Info] Preparation finished after " << timer.elapsed() << " s" << std::endl;
  std::cout << "[Info] Starting training" << std::endl;
  timer.restart();
  for (int iEpoch = 0; iEpoch < epochCount; ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

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
          poshidprobs.data().begin(), sigmoid<float>());

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
          poshidstates.data().begin(), sample_units<float>()
      );

      /*** START NEGATIVE PHASE ***/

      // Calculate p(y | X, H, W) = sigm((X*Wx o H*Wh) * WyT + B)
      HWh = tbblas::prod(poshidstates, Wh);     // recompute using the sampled version of H
      negdata = tbblas::prod(NxF = XWx * HWh, tbblas::trans(Wy));
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(negdata, iRow) += b;

      // For the binary case
      if (!getIsGaussian()) {
        thrust::transform(negdata.begin(), negdata.end(), negdata.begin(),
            sigmoid<float>());
      }

      // Pre-compute Yneg*Wy
      YWy = tbblas::prod(negdata, Wy);

      // Calculate p(h | Yneg, X, W) = sigm((X*Wx o Yneg*Wy) * WhT + C)
      neghidprobs = tbblas::prod((NxF = XWx * YWy), tbblas::trans(Wh));         // x = (X*Wx o Yneg*Wy) * WhT
      for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)                   // x = x + C
        tbblas::row(neghidprobs, iRow) += c;

      thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(),   // x = sigm(x)
          neghidprobs.data().begin(), sigmoid<float>());

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

      float err = 0.f;
      err = tbblas::norm_2(negdata -= ybatch);
      error += err * err;
      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      if (iEpoch) {
        Wxinc = momentum * Wxinc + epsilonw * (((posEx -= negEx) / (float)batchSize) -= weightcost * Wx);
        Wyinc = momentum * Wyinc + epsilonw * (((posEy -= negEy) / (float)batchSize) -= weightcost * Wy);
        Whinc = momentum * Whinc + epsilonw * (((posEh -= negEh) / (float)batchSize) -= weightcost * Wh);
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
    std::cout << "Epoch " << iEpoch << " error " << error << " after " << timer.elapsed() << "s. ETA: "
        << hours << " h " << minutes << " min " << sec << " s" << std::endl;
  }

  data->setFgrbmModel(fgrbm);
}

}

}


