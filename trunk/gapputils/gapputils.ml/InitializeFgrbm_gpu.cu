/*
 * FgrbmTrainer_gpu.cu
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT

#include "InitializeFgrbm.h"

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

void InitializeFgrbm::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  boost::timer timer;

  if (!data)
    data = new InitializeFgrbm();

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

  std::cout << "Initializing FGRBM ..." << std::endl;

//  FgrbmModel test;
//  LOCATE(test, VisibleMean);
//  LOCATE(test, VisibleStd);
//  LOCATE(test, VisibleBiases);
//  LOCATE(test, HiddenBiases);
//  LOCATE(test, VisibleWeights);
//  LOCATE(test, HiddenWeights);
//  LOCATE(test, ConditionalWeights);

  // Calculate the mean and the std of all features
  const unsigned visibleCount = getVisibleCount();
  const unsigned hiddenCount = getHiddenCount();
  const unsigned factorCount = getFactorCount();
  const unsigned sampleCount = getVisiblesVector()->size() / visibleCount;

  boost::shared_ptr<FgrbmModel> fgrbm(new FgrbmModel());
  fgrbm->setIsGaussian(getIsGaussian());

  ublas::matrix<double> visiblesSet(sampleCount, visibleCount);
  ublas::matrix<double> conditionalsSet(sampleCount, visibleCount);
  std::copy(getVisiblesVector()->begin(), getVisiblesVector()->end(), visiblesSet.data().begin());
  std::copy(getConditionalsVector()->begin(), getConditionalsVector()->end(), conditionalsSet.data().begin());

  double mean = thrust::reduce(visiblesSet.data().begin(), visiblesSet.data().end()) / visiblesSet.data().size();
  fgrbm->setVisibleMean(mean);
  std::cout << "[Info] Means calculated: " << timer.elapsed() << " s (" << mean << ")" << std::endl;

  if (getIsGaussian()) {
    thrust::transform(visiblesSet.data().begin(), visiblesSet.data().end(), thrust::constant_iterator<double>(mean),
        visiblesSet.data().begin(), thrust::minus<double>());
    thrust::transform(conditionalsSet.data().begin(), conditionalsSet.data().end(), thrust::constant_iterator<double>(mean),
        conditionalsSet.data().begin(), thrust::minus<double>());

    double stddev = sqrt(thrust::inner_product(visiblesSet.data().begin(), visiblesSet.data().end(),
        visiblesSet.data().begin(), 0.f) / visiblesSet.data().size());
    fgrbm->setVisibleStd(stddev);
    std::cout << "[Info] Standard deviations calculated: " << timer.elapsed() << " s (" << stddev << ")" << std::endl;
  }

  int deviceMemory = 0;
  deviceMemory += 2 * visibleCount * factorCount;
  deviceMemory += 1 * hiddenCount * factorCount;
  deviceMemory += 1 * visibleCount;
  deviceMemory += 1 * hiddenCount;
  std::cout << "[Info] Required device memory without training data: " << sizeof(double) * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  deviceMemory += visiblesSet.size1() * visiblesSet.size2();
  deviceMemory += conditionalsSet.size1() * conditionalsSet.size2();
  std::cout << "[Info] Required device memory including training data: " << sizeof(double) * deviceMemory / 1024. / 1024. << " MB" << std::endl;

  // Initialize weights and bias terms
  boost::shared_ptr<tbblas::device_matrix<double> > conditionalWeights(new tbblas::device_matrix<double>(visibleCount, factorCount));
  boost::shared_ptr<tbblas::device_matrix<double> > visibleWeights(new tbblas::device_matrix<double>(visibleCount, factorCount));
  boost::shared_ptr<tbblas::device_matrix<double> > hiddenWeights(new tbblas::device_matrix<double>(hiddenCount, factorCount));
  boost::shared_ptr<tbblas::device_vector<double> > visibleBiases(new tbblas::device_vector<double>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<double> > hiddenBiases(new tbblas::device_vector<double>(hiddenCount));
  tbblas::device_matrix<double>& Wx = *conditionalWeights;
  tbblas::device_matrix<double>& Wy = *visibleWeights;
  tbblas::device_matrix<double>& Wh = *hiddenWeights;
  tbblas::device_vector<double>& b = *visibleBiases;
  tbblas::device_vector<double>& c = *hiddenBiases;

  // Initialize weights
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wx.data().size()),
      Wx.data().begin(), get_randn<double>(0.f, getWeightStddevs()));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wy.data().size()),
      Wy.data().begin(), get_randn<double>(0.f, getWeightStddevs()));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wh.data().size()),
      Wh.data().begin(), get_randn<double>(0.f, getWeightStddevs()));

  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wx.data().size()),
      Wx.data().begin(), Wx.data().begin(), add_diagonal<double>(Wx.size1(), getDiagonalWeightMeans()));
  thrust::transform(thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(Wy.data().size()),
      Wy.data().begin(), Wy.data().begin(), add_diagonal<double>(Wy.size1(), getDiagonalWeightMeans()));

  // Initialize bias terms
  thrust::fill(b.data().begin(), b.data().end(), 0.f);
  thrust::fill(c.data().begin(), c.data().end(), getInitialHidden());

  std::cout << "[Info] FGRBM initialized: " << timer.elapsed() << " s" << std::endl;

  fgrbm->setConditionalWeights(conditionalWeights);
  fgrbm->setVisibleWeights(visibleWeights);
  fgrbm->setHiddenWeights(hiddenWeights);
  fgrbm->setVisibleBiases(visibleBiases);
  fgrbm->setHiddenBiases(hiddenBiases);

  data->setFgrbmModel(fgrbm);
}

}

}
