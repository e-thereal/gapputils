/*
 * FgrbmEncoder_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include "FgrbmEncoder.h"

#include <capputils/Verifier.h>
#include <algorithm>
#include <cassert>

#include "sampling.hpp"
#include "RbmModel.h"

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void FgrbmEncoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FgrbmEncoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getFgrbmModel() || !getVisibleVector() || !getConditionalVector())
    return;

  FgrbmModel& fgrbm = *getFgrbmModel();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = fgrbm.getVisibleBiases()->size();
  const unsigned factorCount = fgrbm.getVisibleWeights()->size2();

  const unsigned hiddenCount = fgrbm.getHiddenBiases()->size();
  const unsigned sampleCount = getVisibleVector()->size() / visibleCount;

  ublas::matrix<double> visibles(sampleCount, visibleCount), conditionals(sampleCount, visibleCount);
  std::copy(getVisibleVector()->begin(), getVisibleVector()->end(), visibles.data().begin());
  std::copy(getConditionalVector()->begin(), getConditionalVector()->end(), conditionals.data().begin());

  tbblas::device_matrix<double> X(sampleCount, visibleCount), Y(sampleCount, visibleCount);
  X = conditionals;
  Y = visibles;

  // normalize visible variables -> X (design matrix with one sample per row)
  if (getIsGaussian()) {
    thrust::transform(X.data().begin(), X.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleMean()),
        X.data().begin(), thrust::minus<double>());
    thrust::transform(Y.data().begin(), Y.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleMean()),
        Y.data().begin(), thrust::minus<double>());

    thrust::transform(X.data().begin(), X.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleStd()),
        X.data().begin(), thrust::divides<double>());
    thrust::transform(Y.data().begin(), Y.data().end(), thrust::constant_iterator<double>(fgrbm.getVisibleStd()),
        Y.data().begin(), thrust::divides<double>());
  }

  tbblas::device_matrix<double>& Wx = *fgrbm.getConditionalWeights();
  tbblas::device_matrix<double>& Wy = *fgrbm.getVisibleWeights();
  tbblas::device_matrix<double>& Wh = *fgrbm.getHiddenWeights();
  tbblas::device_vector<double>& c = *fgrbm.getHiddenBiases();
  tbblas::device_matrix<double> hidprobs(sampleCount, hiddenCount);

  tbblas::device_matrix<double> XWx(sampleCount, factorCount), YWy(sampleCount, factorCount), NxF(sampleCount, factorCount);

  // Pre-compute X*Wx and Y*Wy
  XWx = tbblas::prod(X, Wx);
  YWy = tbblas::prod(Y, Wy);

  // Calculate p(h | X, Y, W) = sigm((XWx o YWy) * WhT + C)
  hidprobs = tbblas::prod(NxF = XWx * YWy, tbblas::trans(Wh));         // x = (XWx o YWy) * WhT
  for (unsigned iRow = 0; iRow < hidprobs.size1(); ++iRow)             // x = x + C
    tbblas::row(hidprobs, iRow) += c;

  thrust::transform(hidprobs.data().begin(), hidprobs.data().end(), // x = sigm(x)
      hidprobs.data().begin(), sigmoid<double>());
  
  if (getSampleHiddens()) {
    thrust::transform(
        hidprobs.data().begin(), hidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
        hidprobs.data().begin(), sample_units<double>()
    );
  }

  ublas::matrix<double> hiddens = hidprobs;
  boost::shared_ptr<std::vector<double> > hiddenVector(new std::vector<double>(sampleCount * hiddenCount));
  std::copy(hiddens.data().begin(), hiddens.data().end(), hiddenVector->begin());
  data->setHiddenVector(hiddenVector);
}

}

}

