/*
 * Encoder_gpu.cu
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#include "Encoder.h"

#include <tbblas/repeat.hpp>
#include <tbblas/linalg.hpp>
#include <tbblas/math.hpp>

#include "math.hpp"

namespace gml {

namespace rbm {

EncoderChecker::EncoderChecker() {
  Encoder test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Direction, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Encoder::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  Model& rbm = *getModel();
  UnitType visibleUnitType = rbm.getVisibleUnitType();
  UnitType hiddenUnitType = rbm.getHiddenUnitType();

  matrix_t W = *rbm.getWeightMatrix();
  matrix_t b = *rbm.getVisibleBiases();
  matrix_t c = *rbm.getHiddenBiases();

  matrix_t mean = *rbm.getMean();
  matrix_t stddev = *rbm.getStddev();

  std::vector<data_t>& inputs = *getInputs();

  // Calculate the mean and the std of all features
  const size_t visibleCount = W.size()[0];
  const size_t hiddenCount = W.size()[1];
  const size_t sampleCount = inputs.size();

  matrix_t X(sampleCount, inputs[0]->size());
  for (size_t i = 0; i < sampleCount; ++i) {
    thrust::copy(inputs[i]->begin(), inputs[i]->end(), row(X, i).begin());
  }

  if (getDirection() == CodingDirection::Encode) {
    switch (visibleUnitType) {
      case UnitType::Gaussian:
        X = (X - repeat(mean, X.size() / mean.size())) / repeat(stddev, X.size() / stddev.size());
        break;
    }
    matrix_t H;

    // Calculate p(h | X, W) = sigm(XW + C)
    H = prod(X, W);
    H = H + repeat(c, H.size() / c.size());

    switch(hiddenUnitType) {
      case UnitType::Bernoulli: H = sigm(H);    break;
      case UnitType::ReLU:      H = max(0, H);  break;
      case UnitType::MyReLU:    H = nrelu_mean(H); break;
      case UnitType::ReLU1:     H = min(1.0, max(0.0, H));  break;
      case UnitType::ReLU2:     H = min(2.0, max(0.0, H));  break;
      case UnitType::ReLU4:     H = min(4.0, max(0.0, H));  break;
      case UnitType::ReLU8:     H = min(8.0, max(0.0, H));  break;
      default:
        dlog(Severity::Error) << "Hidden unit type '" << hiddenUnitType << "' has not yet been implemented.";
    }

    boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
    for (size_t i = 0; i < sampleCount; ++i) {
      data_t output(new std::vector<double>(hiddenCount));
      thrust::copy(row(H, i).begin(), row(H, i).end(), output->begin());
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  } else {
    // Calculate p(x | H, W) = sigm(HW' + B)
    matrix_t V;
    V = prod(X, tbblas::trans(W));
    V = V + repeat(b, V.size() / b.size());

    switch (visibleUnitType) {
      case UnitType::Bernoulli: V = sigm(V); break;
      case UnitType::Gaussian: break;
      case UnitType::ReLU:      V = max(0, V);  break;
      case UnitType::MyReLU:    V = nrelu_mean(V); break;
      case UnitType::ReLU1:     V = min(1.0, max(0.0, V));  break;
      case UnitType::ReLU2:     V = min(2.0, max(0.0, V));  break;
      case UnitType::ReLU4:     V = min(4.0, max(0.0, V));  break;
      case UnitType::ReLU8:     V = min(8.0, max(0.0, V));  break;
      default:
        dlog(Severity::Error) << "Visible unit type '" << visibleUnitType << "' has not yet been implemented.";
    }

    V = V * repeat(stddev, V.size() / stddev.size()) + repeat(mean, V.size() / mean.size());
    boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
    for (size_t i = 0; i < sampleCount; ++i) {
      data_t output(new std::vector<double>(visibleCount));
      thrust::copy(row(V, i).begin(), row(V, i).end(), output->begin());
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  }
}

}

}
