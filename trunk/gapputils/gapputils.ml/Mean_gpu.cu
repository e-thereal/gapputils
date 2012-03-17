/*
 * Mean_gpu.cu
 *
 *  Created on: Mar 13, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT

#include "Mean.h"

#include <capputils/Verifier.h>

#include <tbblas/device_matrix.hpp>

#include <algorithm>
#include <iostream>

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

void Mean::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  if (!data)
    data = new Mean();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputVectors()) {
    std::cout << "[Warning] No input given." << std::endl;
    return;
  }

  std::vector<float>& inputs = *getInputVectors();

  if (inputs.size() % getFeatureCount()) {
    std::cout << "[Warning] FeatureCount doesn't match total vector size." << std::endl;
    return;
  }

  const int cFeature = getFeatureCount();
  const int cSample = inputs.size() / cFeature;

  ublas::matrix<float> m(cSample, cFeature);
  std::copy(inputs.begin(), inputs.end(), m.data().begin());

  tbblas::device_matrix<float> dm(cSample, cFeature);
  tbblas::device_vector<float> mean(cFeature);
  dm = m;

  mean = tbblas::sum(dm);
  thrust::transform(mean.data().begin(), mean.data().end(), mean.data().begin(), _1 / (float)cSample);

  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(cFeature));
  thrust::copy(mean.data().begin(), mean.data().end(), output->begin());

  data->setOutputVector(output);
}

}

}

