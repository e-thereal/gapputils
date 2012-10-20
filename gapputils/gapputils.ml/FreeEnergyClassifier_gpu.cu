/*
 * FreeEnergyClassifier_gpu.cu
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#include "FreeEnergyClassifier.h"

#include <thrust/inner_product.h>

#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

namespace gapputils {

namespace ml {

struct log1pexp {
__device__ __host__
float operator()(const float& x) const {
  return log(1 + exp(x));
}
};

void FreeEnergyClassifier::update(workflow::IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace thrust::placeholders;

  capputils::Logbook& dlog = getLogbook();

  boost::shared_ptr<std::vector<float> > differences(new std::vector<float>());

  size_t featureCount = getFeatureCount();
  std::vector<float>& conditionals = *getConditionals();
  RbmModel& rbm = *getRbm();

  // Joint vector
  device_matrix<float> x(featureCount + 1, 1);
  device_vector<float>& b = *rbm.getVisibleBiases();
  device_vector<float>& c = *rbm.getHiddenBiases();
  device_matrix<float>& W = *rbm.getWeightMatrix();

  device_matrix<float> WTx(c.size(), 1);
  device_vector<float> e(1);
  float energy;

  if (b.size() != featureCount + 1) {
    dlog(capputils::Severity::Warning) << "Numer of visibles must be equal to the dimension of the conditionals + 1. Abording!";
    return;
  }

  // diff = F(1) - F(0) = -F(0) + F(1) (I calculate -F, therefore, I calculate -F(0) first)

  for (size_t offset  = 0; offset < conditionals.size() && (monitor ? !monitor->getAbortRequested() : 1); offset += featureCount) {
    thrust::copy(conditionals.begin() + offset, conditionals.begin() + offset + featureCount, x.data().begin());

    if (rbm.getIsGaussian()) {
      const float mean = rbm.getVisibleMeans()->data()[0];
      const float stddev = rbm.getVisibleStds()->data()[0];
      thrust::transform(x.data().begin(), x.data().end(), x.data().begin(), (_1 - mean) / stddev);
    }

    if (getMakeBernoulli()) {
      thrust::transform(x.data().begin(), x.data().end(), rbm.getVisibleMeans()->data().begin(),
          x.data().begin(), _1 - _2);
      thrust::transform(x.data().begin(), x.data().end(), rbm.getVisibleStds()->data().begin(),
          x.data().begin(), _1 / _2);
    }

      /*** -F(0) ***/

    x.data()[featureCount] = 0.f;

    // xTb
    energy = thrust::inner_product(x.data().begin(), x.data().end(), b.data().begin(), 0.f);

    // sum(log(1 + exp(WTx + c)))
    WTx = prod(trans(W), x);
    column(WTx, 0) += c;
    thrust::transform(WTx.begin(), WTx.end(), WTx.begin(), log1pexp());
    e = sum(WTx);

    energy += e(0);

      /*** +F(1) ***/

    x.data()[featureCount] = 1.f;

    // xTb
    energy -= thrust::inner_product(x.data().begin(), x.data().end(), b.data().begin(), 0.f);

    // sum(log(1 + exp(WTx + c)))
    WTx = prod(trans(W), x);
    column(WTx, 0) += c;
    thrust::transform(WTx.begin(), WTx.end(), WTx.begin(), log1pexp());
    e = sum(WTx);

    energy -= e(0);

    differences->push_back(energy);
    if (monitor)
      monitor->reportProgress(100. * offset / conditionals.size());
  }

  newState->setDifferences(differences);
}

}

}
