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

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

template<class T>
struct minus_squared : thrust::binary_function<float, float, float> {

T operator()(const T& x, const T& y) const {
  return (x - y) * (x - y);
}

};

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

  boost::shared_ptr<FgrbmModel> fgrbm(new FgrbmModel());

  ublas::matrix<float> visiblesSet(sampleCount, visibleCount);
  ublas::matrix<float> conditionalsSet(sampleCount, visibleCount);
  std::copy(getVisiblesVector()->begin(), getVisiblesVector()->end(), visiblesSet.data().begin());
  std::copy(getConditionalsVector()->begin(), getConditionalsVector()->end(), conditionalsSet.data().begin());

  float mean = thrust::reduce(visiblesSet.data().begin(), visiblesSet.data().end()) / visiblesSet.data().size();
  fgrbm->setVisibleMean(mean);
  std::cout << "[Info] Means calculated: " << timer.elapsed() << " s" << std::endl;

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
  ublas::matrix<float>& uX = conditionalsSet;
  ublas::matrix<float>& uY = visiblesSet;

  for (unsigned i = uX.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    ublas::row(uX, i).swap(ublas::row(uX, j));
    ublas::row(uY, i).swap(ublas::row(uY, j));
  }
  std::cout << "[Info] Rows shuffled: " << timer.elapsed() << " s" << std::endl;

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
      Wx.data().begin(), get_randn<float>(0.f, 0.1f));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wy.data().size()),
      Wy.data().begin(), get_randn<float>(0.f, 0.1f));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wh.data().size()),
      Wh.data().begin(), get_randn<float>(0.f, 0.1f));

  // Initialize bias terms
  thrust::fill(b.data().begin(), b.data().end(), 0.f);
  thrust::fill(c.data().begin(), c.data().end(), getInitialHidden());
  std::cout << "[Info] RBM initialized: " << timer.elapsed() << " s" << std::endl;

  fgrbm->setConditionalWeights(conditionalWeights);
  fgrbm->setVisibleWeights(visibleWeights);
  fgrbm->setHiddenWeights(hiddenWeights);
  fgrbm->setVisibleBiases(visibleBiases);
  fgrbm->setHiddenBiases(hiddenBiases);

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

  tbblas::device_matrix<float> xbatch(batchSize, visibleCount);
  tbblas::device_matrix<float> ybatch(batchSize, visibleCount);
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
  tbblas::device_matrix<float> Whinc(visibleCount, factorCount);

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

//      // Get current batch
//      batch = tbblas::subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());
//      //read_matrix("data.bin", batch);
//
//      // Calculate p(h | X, W) = sigm(XW + C)
//      REPEAT poshidprobs = tbblas::prod(batch, W);
//      TOC
//      REPEAT
//      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)
//        tbblas::row(poshidprobs, iRow) += c;
//      TOC
//      REPEAT thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
//          poshidprobs.data().begin(), sigmoid<float>());
//      TOC
//
//      // (x_n)(mu_n)'
//      TIC
//      REPEAT posprods = tbblas::prod(tbblas::trans(batch), poshidprobs);
//      TOC
//
//      TIC
//      // Calculate the total activation of the hidden and visible units
//      REPEAT
//        poshidact = tbblas::sum(poshidprobs);
//      TOC
//
//      TIC
//      REPEAT
//        posvisact = tbblas::sum(batch);
//      TOC
//
//      /*tbblas::device_matrix<float> posprods_test;
//      tbblas::device_vector<float> poshidact_test, posvisact_test;
//      read_matrix("posprods.bin", posprods_test);
//      read_vector("poshidact.bin", poshidact_test);
//      read_vector("posvisact.bin", posvisact_test);
//
//      std::cout << "Positive phase errors:" << std::endl;
//      std::cout << tbblas::norm_1(posprods_test -= posprods) / posprods.data().size() << std::endl;
//      std::cout << tbblas::norm_1(poshidact_test -= poshidact) / poshidact.size() << std::endl;
//      std::cout << tbblas::norm_1(posvisact_test -= posvisact) / posvisact.size() << std:: endl;*/
//
//      /*** END OF POSITIVE PHASE ***/
//
//      // Sample the hidden states
//      TIC
//      REPEAT thrust::transform(
//          poshidprobs.data().begin(), poshidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
//          poshidstates.data().begin(), sample_units<float>()
//      );
//      TOC
//
//      //std::cout << "      E[p(h)] = " << ublas::norm_1(poshidprobs) / poshidprobs.data().size() << std::endl;
//      //std::cout << " E[h] (ublas) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;
//
//      //read_matrix("poshidstates.bin", poshidstates);
//      //std::cout << "E[h] (matlab) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;
//
//      /*** START NEGATIVE PHASE ***/
//
//      // Calculate p(x | H, W) = sigm(HW' + B)
//      TIC
//      REPEAT negdata = tbblas::prod(poshidstates, tbblas::trans(W));
//      TOC
//      REPEAT for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
//        tbblas::row(negdata, iRow) += b;
//      TOC
//
//      // For the binary case
//      if (!getIsGaussian()) {
//        REPEAT
//        thrust::transform(negdata.begin(), negdata.end(), negdata.begin(),
//            sigmoid<float>());
//        TOC
//      }
//
//      // Calculate p(h | Xneg, W) = sigm(XnegW + C)
//      TIC
//      REPEAT
//      neghidprobs = tbblas::prod(negdata, W);
//      TOC
//      REPEAT for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
//        tbblas::row(neghidprobs, iRow) += c;
//      TOC
//
//      REPEAT
//      thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(), neghidprobs.data().begin(),
//          sigmoid<float>());
//      TOC
//
//      // (xneg)(mu_neg)'
//      TIC
//      REPEAT
//      negprods = tbblas::prod(tbblas::trans(negdata), neghidprobs);
//      TOC
//
//      // Calculate the total activation of the visible and hidden units (reconstruction)
//      TIC
//      REPEAT
//      neghidact = tbblas::sum(neghidprobs);
//      TOC
//      TIC
//      REPEAT
//      negvisact = tbblas::sum(negdata);
//      TOC
//
//      /*tbblas::device_matrix<float> negprods_test;
//      tbblas::device_vector<float> neghidact_test, negvisact_test;
//      read_matrix("negprods.bin", negprods_test);
//      read_vector("neghidact.bin", neghidact_test);
//      read_vector("negvisact.bin", negvisact_test);
//
//      std::cout << "Negative phase errors:" << std::endl;
//      std::cout << tbblas::norm_1(negprods_test -= negprods) / negprods.data().size() << std::endl;
//      std::cout << tbblas::norm_1(neghidact_test -= neghidact) / neghidact.size() << std::endl;
//      std::cout << tbblas::norm_1(negvisact_test -= negvisact) / negvisact.size() << std:: endl;*/
//
//      /*** END OF NEGATIVE PHASE ***/
//
//      if (iEpoch == epochCount - 1) {
//        ublas::matrix<float> temp = negdata;
//        thrust::copy(temp.data().begin(), temp.data().end(), debugNegData->begin() + (iBatch * batchSize * visibleCount));
//      }
//
//      float err = 0.f;
//      TIC REPEAT
//      err = tbblas::norm_2(negdata -= batch);
//      TOC
//      error += err * err;
//      //std::cout << "norm2(negdata) = " << tbblas::norm_2(negdata) << std::endl;
//      //std::cout << "norm2(data)    = " << tbblas::norm_2(batch) << std::endl;
//      //std::cout << "Epoch " << iEpoch << " batch " << iBatch << " error " << err * err << std::endl;
//
//      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);
//
//      /*** UPDATE WEIGHTS AND BIASES ***/
//
//      //if (iEpoch) {
//        // Don't learn anything in the first epoch in order to get a good estimate of the initial error
//      TIC REPEAT
//        vishidinc = momentum * vishidinc + epsilonw * (((posprods -= negprods) / (float)batchSize) -= weightcost * W);
//      TOC
//      REPEAT
//        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact -= negvisact);
//      TOC
//      REPEAT
//        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact -= neghidact);
//      TOC
//
//      //}
//
//      /*tbblas::device_matrix<float> vishidinc_test;
//      tbblas::device_vector<float> visbiasinc_test, hidbiasinc_test;
//      read_matrix("vishidinc.bin", vishidinc_test);
//      read_vector("visbiasinc.bin", visbiasinc_test);
//      read_vector("hidbiasinc.bin", hidbiasinc_test);
//
//      std::cout << "Finalization phase errors:" << std::endl;
//      std::cout << tbblas::norm_1(vishidinc_test -= vishidinc) / vishidinc.data().size() << std::endl;
//      std::cout << tbblas::norm_1(visbiasinc_test -= visbiasinc) / visbiasinc.size() << std::endl;
//      std::cout << tbblas::norm_1(hidbiasinc_test -= hidbiasinc) / hidbiasinc.size() << std:: endl;*/
//
//      TIC REPEAT
//      W += vishidinc;
//      TOC REPEAT
//      b += visbiasinc;
//      TOC REPEAT
//      c += hidbiasinc;
//      TOC

      /*** END OF UPDATES ***/

      //std::cout << "Time: " << timer.elapsed() << "s" << std::endl;

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


