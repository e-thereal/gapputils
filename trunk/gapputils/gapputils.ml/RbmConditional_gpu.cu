/*
 * RbmConditional_gpu.cu
 *
 *  Created on: Mar 12, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT
#include "RbmConditional.h"

#include <capputils/Verifier.h>
#include <algorithm>

#include "sampling.hpp"
#include <curand.h>
#include <cmath>

#include <boost/thread/thread.hpp>

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void RbmConditional::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  namespace tl = thrust::placeholders;

  if (!data)
    data = new RbmConditional();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getModel()) {
    std::cout << "[Warning] Invalid inputs." << std::endl;
    return;
  }

  // Initialize: joint vector, encoded vector, mean conditional vector
  RbmModel& rbm = *getModel();

  if (rbm.getIsGaussian()) {
    std::cout << "[Warning] RBM model must not be Gaussian-Bernoulli." << std::endl;
  }

  const int cVisible = rbm.getVisibleBiases()->size();
  const int cHidden = rbm.getHiddenBiases()->size();

  bool isConditional = getGivens() && (getGivenCount() != 0);
  const int cGiven = isConditional ? getGivenCount() : 0;

  if (isConditional && (cGiven >= cVisible || getGivens()->size() % cGiven)) {
    std::cout << "[Warning] Invalid number of given units." << std::endl;
    return;
  }

  curandGenerator_t gen;
  curandStatus_t status;
  if ((status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)) != CURAND_STATUS_SUCCESS) {
    std::cout << "[Warning] Could not create random number generator: " << status << std::endl;
    return;
  }

  int cSample = isConditional ? getGivens()->size() / cGiven : 1;

  const int cConditional = cVisible - cGiven;

  if (getDebug()) {
    std::cout << "cSample: " << cSample << std::endl;
    std::cout << "cConditional: " << cConditional << std::endl;
  }

  const int cGenSample = getShowSamples() ? 1 : getSampleCycles();

  boost::shared_ptr<std::vector<float> > conditionals(new std::vector<float>(cGenSample * cSample * cConditional));

  tbblas::device_matrix<float>& W = *rbm.getWeightMatrix();
  tbblas::device_vector<float>& b = *rbm.getVisibleBiases();
  tbblas::device_vector<float>& c = *rbm.getHiddenBiases();
  tbblas::device_matrix<float> H(cSample, cHidden);
  tbblas::device_matrix<float> xGiven(cSample, cGiven),
      xConditional(cSample, cConditional), X(cSample, cVisible);

  thrust::fill(xConditional.data().begin(), xConditional.data().end(), 0.f);

  int cVisibleRand = cSample * cVisible + ((cSample * cVisible) % 2);
  int cHiddenRand = cSample * cHidden + ((cSample * cHidden) % 2);
  thrust::device_vector<float> randomValues(std::max(cVisibleRand, cHiddenRand));

  if ((status = curandGenerateNormal(gen, X.data().data().get(), X.data().size(), 0.f, 0.f))
      != CURAND_STATUS_SUCCESS)
  {
    std::cout << "[Warning] Could not generate random numbers: " << status << std::endl;
  }
//  thrust::transform(X.data().begin(), X.data().end(),
//      X.data().begin(), sigmoid<float>());

  ublas::matrix<float> givens(cSample, cGiven);
  if (isConditional)
    std::copy(getGivens()->begin(), getGivens()->end(), givens.data().begin());
  xGiven = givens;

  data->setConditionals(conditionals);

  for (int i = 0; i < getInitializationCycles() + getSampleCycles() && (monitor ? !monitor->getAbortRequested() : true); ++i) {

    /*** Reset givens ***/

    thrust::copy(xGiven.data().begin(), xGiven.data().end(),
        tbblas::subrange(X, 0,cSample, 0,cGiven).begin());

    /*** Sample encoded joint vector ***/

    // Calculate p(h | X, W) = sigm(XW + C) (here X is the joint)
    H = tbblas::prod(X, W);
    if (getDebug())
      std::cout << "Hidden activation: " << tbblas::norm_1(H);
    for (unsigned iRow = 0; iRow < H.size1(); ++iRow)
      tbblas::row(H, iRow) += c;

    if (getDebug())
      std::cout << ", biased = " << tbblas::norm_1(H);

    thrust::transform(H.data().begin(), H.data().end(),
        H.data().begin(), sigmoid<float>());

    if ((status = curandGenerateUniform(gen,
        randomValues.data().get(),
        cHiddenRand)) != CURAND_STATUS_SUCCESS)
    {
      std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
      return;
    }

    if (getDebug())
      std::cout << ", p = " << tbblas::norm_1(H);

    thrust::transform(
        H.data().begin(), H.data().end(), randomValues.begin(),
        H.data().begin(), tl::_1 > tl::_2
    );

    if (getDebug())
      std::cout << ", s = " << tbblas::norm_1(H) << std::endl;

    /*** Sample decoded joint vector ***/

    // Calculate p(x | H, W) = sigm(HW' + B)
    X = tbblas::prod(H, tbblas::trans(W));
    for (unsigned iRow = 0; iRow < X.size1(); ++iRow)
      tbblas::row(X, iRow) += b;

    if (!rbm.getIsGaussian()) {
      thrust::transform(X.data().begin(), X.data().end(), X.data().begin(),
          sigmoid<float>());
    }

    if (i >= getInitializationCycles() && getShowSamples()) {
      tbblas::device_matrix<float> xConditionalPart = tbblas::subrange(X, 0,cSample, cGiven,cVisible);
      thrust::copy(xConditionalPart.begin(), xConditionalPart.end(), conditionals->begin());
      if (monitor)
        monitor->reportProgress(100. * i / (getInitializationCycles() + getSampleCycles()), true);
      if (getDelay())
        boost::this_thread::sleep(boost::posix_time::milliseconds(getDelay()));
    }

    if ((status = curandGenerateUniform(gen,
        randomValues.data().get(),
        cVisibleRand)) != CURAND_STATUS_SUCCESS)
    {
      std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
      return;
    }

    if (!rbm.getIsGaussian()) {
      thrust::transform(
          X.data().begin(), X.data().end(), randomValues.begin(),
          X.data().begin(), tl::_1 > tl::_2
      );
    }

    if (i >= getInitializationCycles()) {

      /*** Accumulate conditionals ***/

      if (!getShowSamples()) {
//        xConditional += tbblas::subrange(X, 0,cSample, cGiven,cVisible);
        tbblas::device_matrix<float> xConditionalPart = tbblas::subrange(X, 0,cSample, cGiven,cVisible);
        thrust::copy(xConditionalPart.begin(), xConditionalPart.end(),
            conditionals->begin() + (i - getInitializationCycles()) * cSample * cConditional);
        if (monitor)
          monitor->reportProgress(100. * i / (getInitializationCycles() + getSampleCycles()));
      }
    }
  }

  if (!getShowSamples()) {
//    xConditional = xConditional / (float)getSampleCycles();
//    thrust::copy(xConditional.data().begin(), xConditional.data().end(), conditionals->begin());
  }

  if ((status = curandDestroyGenerator(gen)) != CURAND_STATUS_SUCCESS)
  {
    std::cout << "[Error] Could not destroy random number generator: " << status << std::endl;
    return;
  }
}

}

}

