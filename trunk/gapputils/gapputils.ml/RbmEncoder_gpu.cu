/*
 * RbmEncoder_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */

#include "RbmEncoder.h"

#include <capputils/Verifier.h>
#include <algorithm>

#include "sampling.hpp"

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void RbmEncoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmEncoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getVisibleVector())
    return;

  RbmModel& rbm = *getRbmModel();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = rbm.getVisibleBiases()->size();
  const unsigned hiddenCount = rbm.getHiddenBiases()->size();
  const unsigned sampleCount = getVisibleVector()->size() / visibleCount;

  ublas::matrix<float> visibles(sampleCount, visibleCount);
  std::copy(getVisibleVector()->begin(), getVisibleVector()->end(), visibles.data().begin());

  // normalize visible variables -> X (design matrix with one sample per row)
  boost::shared_ptr<ublas::matrix<float> > normalizedVisibles = getRbmModel()->encodeDesignMatrix(visibles, !getIsGaussian());
  tbblas::device_matrix<float> X(sampleCount, visibleCount);
  X = *normalizedVisibles;

  tbblas::device_matrix<float>& W = *rbm.getWeightMatrix();
  tbblas::device_vector<float>& c = *rbm.getHiddenBiases();
  tbblas::device_matrix<float> hidprobs(sampleCount, hiddenCount);

  // Calculate p(h | X, W) = sigm(XW + C)
  hidprobs = tbblas::prod(X, W);
  for (unsigned iRow = 0; iRow < hidprobs.size1(); ++iRow)
    tbblas::row(hidprobs, iRow) += c;

  thrust::transform(hidprobs.data().begin(), hidprobs.data().end(),
      hidprobs.data().begin(), sigmoid<float>());

  if (getSampleHiddens()) {
    thrust::transform(
        hidprobs.data().begin(), hidprobs.data().end(), thrust::counting_iterator<unsigned>(0),
        hidprobs.data().begin(), sample_units<float>()
    );
  }
  ublas::matrix<float> hiddens = hidprobs;
  boost::shared_ptr<std::vector<float> > hiddenVector(new std::vector<float>(sampleCount * hiddenCount));
  std::copy(hiddens.data().begin(), hiddens.data().end(), hiddenVector->begin());
  data->setHiddenVector(hiddenVector);
}

}

}

