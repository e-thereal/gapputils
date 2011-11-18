/*
 * RbmDecoder_gpu.cu
 *
 *  Created on: Nov 16, 2011
 *      Author: tombr
 */

#include "RbmDecoder.h"

#include <capputils/Verifier.h>

#include "sampling.hpp"

namespace gapputils {

namespace ml {

namespace ublas = boost::numeric::ublas;

void RbmDecoder::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmDecoder();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getRbmModel() || !getHiddenVector())
      return;

  RbmModel& rbm = *getRbmModel();

  // Calculate the mean and the std of all features
  const unsigned visibleCount = rbm.getVisibleBiases()->size();
  const unsigned hiddenCount = rbm.getHiddenBiases()->size();
  const unsigned sampleCount = getHiddenVector()->size() / hiddenCount;

  ublas::matrix<float> hiddens(sampleCount, hiddenCount);
  std::copy(getHiddenVector()->begin(), getHiddenVector()->end(), hiddens.data().begin());

  tbblas::device_matrix<float> H(sampleCount, hiddenCount);
  H = hiddens;
  tbblas::device_matrix<float>& W = *rbm.getWeightMatrix();
  tbblas::device_vector<float>& b = *rbm.getVisibleBiases();
  tbblas::device_matrix<float> negdata(sampleCount, visibleCount);

  // Calculate p(x | H, W) = sigm(HW' + B)
  negdata = tbblas::prod(H, tbblas::trans(W));
  for (unsigned iRow = 0; iRow < negdata.size1(); ++iRow)
    tbblas::row(negdata, iRow) += b;

  // For the binary case
  if (!getIsGaussian())
    thrust::transform(negdata.data().begin(), negdata.data().end(), negdata.data().begin(),
              sigmoid<float>());

  ublas::matrix<float> visibles(sampleCount, visibleCount);
  visibles = negdata;

  boost::shared_ptr<std::vector<float> > visibleVector(new std::vector<float>(sampleCount * visibleCount));
  if (!getIsGaussian()) {
    std::copy(visibles.data().begin(), visibles.data().end(), visibleVector->begin());
  } else {
    boost::shared_ptr<ublas::matrix<float> > decoded = rbm.decodeApproximation(visibles);
    std::copy(decoded->data().begin(), decoded->data().end(), visibleVector->begin());
  }

  // Decode approximation (reconstruction)
  data->setVisibleVector(visibleVector);
}

}

}
