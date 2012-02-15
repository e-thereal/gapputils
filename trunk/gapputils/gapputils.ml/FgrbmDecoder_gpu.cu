/*
 * FgrbmDecoder_gpu.cu
 *
 *  Created on: Jan 10, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT
#include "FgrbmDecoder.h"

#include <capputils/Verifier.h>
#include <algorithm>
#include <cassert>

#include "sampling.hpp"
#include "RbmModel.h"

#include <iostream>

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void FgrbmDecoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FgrbmDecoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getFgrbmModel() || !getHiddenVector() || !getConditionalVector())
    return;

  FgrbmModel& fgrbm = *getFgrbmModel();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = fgrbm.getVisibleBiases()->size();
  const unsigned factorCount = fgrbm.getVisibleWeights()->size2();

  const unsigned hiddenCount = fgrbm.getHiddenBiases()->size();
  const unsigned sampleCount = getHiddenVector()->size() / hiddenCount;

  ublas::matrix<double> hiddens(sampleCount, hiddenCount), conditionals(sampleCount, visibleCount);
  std::copy(getHiddenVector()->begin(), getHiddenVector()->end(), hiddens.data().begin());
  std::copy(getConditionalVector()->begin(), getConditionalVector()->end(), conditionals.data().begin());

  // normalize visible variables -> X (design matrix with one sample per row)
  tbblas::device_matrix<double> X(sampleCount, visibleCount), H(sampleCount, hiddenCount);
  X = conditionals;
  H = hiddens;

  if (fgrbm.getIsGaussian()) {
    std::cout << "[Info] Normalizing conditionals with mean = " << fgrbm.getVisibleMean() << " and stddev = " << fgrbm.getVisibleStd() << std::endl;
    thrust::transform(X.data().begin(), X.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleMean()),
        X.data().begin(), thrust::minus<double>());

    thrust::transform(X.data().begin(), X.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleStd()),
        X.data().begin(), thrust::divides<double>());
  }

  tbblas::device_matrix<double>& Wx = *fgrbm.getConditionalWeights();
  tbblas::device_matrix<double>& Wy = *fgrbm.getVisibleWeights();
  tbblas::device_matrix<double>& Wh = *fgrbm.getHiddenWeights();
  tbblas::device_vector<double>& b = *fgrbm.getVisibleBiases();
  tbblas::device_matrix<double> negdata(sampleCount, visibleCount);

  tbblas::device_matrix<double> XWx(sampleCount, factorCount), HWh(sampleCount, factorCount), NxF(sampleCount, factorCount);

  // Pre-compute X*Wx and Y*Wy
  XWx = tbblas::prod(X, Wx);
  HWh = tbblas::prod(H, Wh);

  // Calculate p(y | X, H, W) = sigm((X*Wx o H*Wh) * WyT + B)
  negdata = tbblas::prod(NxF = XWx * HWh, tbblas::trans(Wy));
  for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
    tbblas::row(negdata, iRow) += b;

  if (!fgrbm.getIsGaussian()) {
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
    thrust::transform(negdata.data().begin(), negdata.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleStd()),
        negdata.data().begin(), thrust::multiplies<double>());

    thrust::transform(negdata.data().begin(), negdata.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleMean()),
        negdata.data().begin(), thrust::plus<double>());
  }

  ublas::matrix<double> visibles = negdata;
  boost::shared_ptr<std::vector<double> > visibleVector(new std::vector<double>(sampleCount * visibleCount));
  std::copy(visibles.data().begin(), visibles.data().end(), visibleVector->begin());
  data->setVisibleVector(visibleVector);
}

}

}
