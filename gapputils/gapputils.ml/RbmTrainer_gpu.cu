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

#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

#include "tbblas_io.hpp"
#include "sampling.hpp"

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

//#define TIC timer.restart();
//#define TOC cudaThreadSynchronize(); std::cout << __LINE__ << ": " << timer.elapsed() << "s" << std::endl;
//#define REPEAT for(int i = 0; i < 1000; ++i)
#define TIC
#define TOC
#define REPEAT

void RbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  //using namespace boost::lambda;
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
  std::cout << "[Info] Means calculated: " << timer.elapsed() << " s" << std::endl;

  ublas::vector<float>& stds = *visibleStds;
  for (unsigned iCol = 0; iCol < trainingSet.size2(); ++iCol)
    stds(iCol) = ublas::norm_2(ublas::column(trainingSet, iCol) -
      ublas::scalar_vector<float>(trainingSet.size1(), means(iCol)) / trainingSet.size1());
  rbm->setVisibleStds(visibleStds);
  std::cout << "[Info] Standard deviations calculated: " << timer.elapsed() << " s" << std::endl;

  // Apply feature scaling to training set
  boost::shared_ptr<ublas::matrix<float> > scaledSet = rbm->encodeDesignMatrix(trainingSet, !getIsGaussian());
  std::cout << "[Info] Design matrix standardized: " << timer.elapsed() << " s" << std::endl;
  ublas::matrix<float>& uX = *scaledSet;

  for (unsigned i = uX.size1() - 1; i > 0; --i) {
    unsigned j = rand() % (i + 1);
    ublas::row(uX, i).swap(ublas::row(uX, j));
  }
  std::cout << "[Info] Rows shuffled: " << timer.elapsed() << " s" << std::endl;

  tbblas::device_matrix<float> X(scaledSet->size1(), scaledSet->size2());
  std::cout << "[Info] Design matrix allocated: " << timer.elapsed() << " s" << std::endl;
  X = uX;
  std::cout << "[Info] Design matrix written to the device: " << timer.elapsed() << " s" << std::endl;

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
  thrust::fill(c.data().begin(), c.data().end(), 0.f);
  std::cout << "[Info] RBM initialized: " << timer.elapsed() << " s" << std::endl;

  rbm->setWeightMatrix(weightMatrix);
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

  thrust::fill(vishidinc.data().begin(), vishidinc.data().end(), 0.f);
  thrust::fill(hidbiasinc.data().begin(), hidbiasinc.data().end(), 0.f);
  thrust::fill(visbiasinc.data().begin(), visbiasinc.data().end(), 0.f);

  const int epochCount = getEpochCount();

  //read_matrix("data.bin", batch);

  //boost::progress_timer progresstimer;
  std::cout << "[Info] Preparation finished after " << timer.elapsed() << " s" << std::endl;
  std::cout << "[Info] Starting training" << std::endl;
  timer.restart();
  for (int iEpoch = 0; iEpoch < epochCount; ++iEpoch) {

    float error = 0;
    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {
      //std::cout.precision(10);
      TIC
      /*** START POSITIVE PHASE ***/

      // Get current batch
      batch = tbblas::subrange(X, iBatch * batchSize, (iBatch + 1) * batchSize, 0, X.size2());
      //read_matrix("data.bin", batch);

      // Calculate p(h | X, W) = sigm(XW + C)
      REPEAT poshidprobs = tbblas::prod(batch, W);
      TOC
      REPEAT
      for (unsigned iRow = 0; iRow < poshidprobs.size1(); ++iRow)
        tbblas::row(poshidprobs, iRow) += c;
      TOC
      REPEAT thrust::transform(poshidprobs.data().begin(), poshidprobs.data().end(),
          poshidprobs.data().begin(), sigmoid<float>());
      TOC

      // (x_n)(mu_n)'
      TIC
      REPEAT posprods = tbblas::prod(tbblas::trans(batch), poshidprobs);
      TOC

      TIC
      // Calculate the total activation of the hidden and visible units
      // TODO: Sums are way to slow. Calculate all sums simulaniously
      REPEAT // for (unsigned iCol = 0; iCol < poshidprobs.size2(); ++iCol)
        poshidact = tbblas::sum(poshidprobs); //poshidact(iCol) = tbblas::sum(tbblas::column(poshidprobs, iCol));
      TOC

      TIC
      REPEAT // for (unsigned iCol = 0; iCol < batch.size2(); ++iCol)
        posvisact = tbblas::sum(batch); // posvisact(iCol) = tbblas::sum(tbblas::column(batch, iCol));
      TOC

      /*tbblas::device_matrix<float> posprods_test;
      tbblas::device_vector<float> poshidact_test, posvisact_test;
      read_matrix("posprods.bin", posprods_test);
      read_vector("poshidact.bin", poshidact_test);
      read_vector("posvisact.bin", posvisact_test);

      std::cout << "Positive phase errors:" << std::endl;
      std::cout << tbblas::norm_1(posprods_test -= posprods) / posprods.data().size() << std::endl;
      std::cout << tbblas::norm_1(poshidact_test -= poshidact) / poshidact.size() << std::endl;
      std::cout << tbblas::norm_1(posvisact_test -= posvisact) / posvisact.size() << std:: endl;*/

      /*** END OF POSITIVE PHASE ***/

      // Sample the hidden states
      TIC
      REPEAT thrust::transform(
          poshidprobs.data().begin(), poshidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
          poshidstates.data().begin(), sample_units<float>()
      );
      TOC

      //std::cout << "      E[p(h)] = " << ublas::norm_1(poshidprobs) / poshidprobs.data().size() << std::endl;
      //std::cout << " E[h] (ublas) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;

      //read_matrix("poshidstates.bin", poshidstates);
      //std::cout << "E[h] (matlab) = " << ublas::norm_1(poshidstates) / poshidstates.data().size() << std::endl;

      /*** START NEGATIVE PHASE ***/

      // Calculate p(x | H, W) = sigm(HW' + B)
      TIC
      REPEAT negdata = tbblas::prod(poshidstates, tbblas::trans(W));
      TOC
      REPEAT for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(negdata, iRow) += b;
      TOC

      // For the binary case
      if (!getIsGaussian())
        REPEAT
        thrust::transform(negdata.begin(), negdata.end(), negdata.begin(),
            sigmoid<float>());
      TOC

      // Calculate p(h | Xneg, W) = sigm(XnegW + C)
      TIC
      REPEAT
      neghidprobs = tbblas::prod(negdata, W);
      TOC
      REPEAT for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
        tbblas::row(neghidprobs, iRow) += c;
      TOC

      REPEAT
      thrust::transform(neghidprobs.data().begin(), neghidprobs.data().end(), neghidprobs.data().begin(),
          sigmoid<float>());
      TOC

      // (xneg)(mu_neg)'
      TIC
      REPEAT
      negprods = tbblas::prod(tbblas::trans(negdata), neghidprobs);
      TOC

      // Calculate the total activation of the visible and hidden units (reconstruction)
      // TODO: Sums are to slow. Speed up using simultaneous sums
      TIC
      REPEAT
      //for (unsigned iCol = 0; iCol < neghidprobs.size2(); ++iCol)
      neghidact = tbblas::sum(neghidprobs); //  neghidact(iCol) = tbblas::sum(tbblas::column(neghidprobs, iCol));
      TOC
      TIC
      REPEAT
      // for (unsigned iCol = 0; iCol < negdata.size2(); ++iCol)
      negvisact = tbblas::sum(negdata); //  negvisact(iCol) = tbblas::sum(tbblas::column(negdata, iCol));
      TOC

      /*tbblas::device_matrix<float> negprods_test;
      tbblas::device_vector<float> neghidact_test, negvisact_test;
      read_matrix("negprods.bin", negprods_test);
      read_vector("neghidact.bin", neghidact_test);
      read_vector("negvisact.bin", negvisact_test);

      std::cout << "Negative phase errors:" << std::endl;
      std::cout << tbblas::norm_1(negprods_test -= negprods) / negprods.data().size() << std::endl;
      std::cout << tbblas::norm_1(neghidact_test -= neghidact) / neghidact.size() << std::endl;
      std::cout << tbblas::norm_1(negvisact_test -= negvisact) / negvisact.size() << std:: endl;*/

      /*** END OF NEGATIVE PHASE ***/

      float err = 0.f;
      TIC REPEAT
      err = tbblas::norm_2(negdata -= batch);
      TOC
      error += err * err;
      //std::cout << "Epoch " << iEpoch << " batch " << iBatch << " error " << err * err << std::endl;

      momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

      /*** UPDATE WEIGHTS AND BIASES ***/

      //if (iEpoch) {
        // Don't learn anything in the first epoch in order to get a good estimate of the initial error
      // TODO: Investigate, why this is so slow. Maybe optimized axpy for matrices
      TIC REPEAT
        vishidinc = momentum * vishidinc + epsilonw * (((posprods -= negprods) / (float)batchSize) -= weightcost * W);
      TOC
      REPEAT
        visbiasinc = momentum * visbiasinc + (epsilonvb / batchSize) * (posvisact -= negvisact);
      TOC
      REPEAT
        hidbiasinc = momentum * hidbiasinc + (epsilonhb / batchSize) * (poshidact -= neghidact);
      TOC

      //}

      /*tbblas::device_matrix<float> vishidinc_test;
      tbblas::device_vector<float> visbiasinc_test, hidbiasinc_test;
      read_matrix("vishidinc.bin", vishidinc_test);
      read_vector("visbiasinc.bin", visbiasinc_test);
      read_vector("hidbiasinc.bin", hidbiasinc_test);

      std::cout << "Finalization phase errors:" << std::endl;
      std::cout << tbblas::norm_1(vishidinc_test -= vishidinc) / vishidinc.data().size() << std::endl;
      std::cout << tbblas::norm_1(visbiasinc_test -= visbiasinc) / visbiasinc.size() << std::endl;
      std::cout << tbblas::norm_1(hidbiasinc_test -= hidbiasinc) / hidbiasinc.size() << std:: endl;*/

      // TODO: Need optimized axpy for matrices
      TIC REPEAT
      W += vishidinc;
      TOC REPEAT
      b += visbiasinc;
      TOC REPEAT
      c += hidbiasinc;
      TOC

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

  data->setRbmModel(rbm);
}

}

}
