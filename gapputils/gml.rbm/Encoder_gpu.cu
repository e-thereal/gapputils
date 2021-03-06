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

#include <tbblas/deeplearn/rbm.hpp>

namespace gml {

namespace rbm {

EncoderChecker::EncoderChecker() {
  Encoder test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Direction, test);
  CHECK_MEMORY_LAYOUT2(DoubleWeights, test);
  CHECK_MEMORY_LAYOUT2(OnlyFilters, test);
  CHECK_MEMORY_LAYOUT2(NormalizeOnly, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Encoder::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();

#if 1
  model_t& model = *getModel();
  tbblas::deeplearn::rbm<value_t> rbm(model);

  std::vector<data_t>& inputs = *getInputs();

  // Calculate the mean and the std of all features
  const size_t visibleCount = model.weights().size()[0];
  const size_t hiddenCount = model.weights().size()[1];
  const size_t sampleCount = inputs.size();

  if (getDirection() == CodingDirection::Encode) {
    rbm.visibles().resize(seq(sampleCount, visibleCount));
    for (size_t i = 0; i < sampleCount; ++i) {
      thrust::copy(inputs[i]->begin(), inputs[i]->end(), row(rbm.visibles(), i).begin());
    }

    rbm.normalize_visibles();

    if (getNormalizeOnly()) {
      boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
      for (size_t i = 0; i < sampleCount; ++i) {
        data_t output(new std::vector<double>(visibleCount));
        thrust::copy(row(rbm.visibles(), i).begin(), row(rbm.visibles(), i).end(), output->begin());
        outputs->push_back(output);
      }
      newState->setOutputs(outputs);
      return;
    }

    rbm.infer_hiddens();

    boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
    for (size_t i = 0; i < sampleCount; ++i) {
      data_t output(new std::vector<double>(hiddenCount));
      thrust::copy(row(rbm.hiddens(), i).begin(), row(rbm.hiddens(), i).end(), output->begin());
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  } else {
    rbm.hiddens().resize(seq(sampleCount, hiddenCount));
    for (size_t i = 0; i < sampleCount; ++i) {
      thrust::copy(inputs[i]->begin(), inputs[i]->end(), row(rbm.hiddens(), i).begin());
    }

    rbm.infer_visibles(getOnlyFilters());

    if (!getOnlyFilters())
      rbm.diversify_visibles();

    boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
    for (size_t i = 0; i < sampleCount; ++i) {
      data_t output(new std::vector<double>(visibleCount));
      thrust::copy(row(rbm.visibles(), i).begin(), row(rbm.visibles(), i).end(), output->begin());
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  }
#else
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

    if (getNormalizeOnly()) {
      boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
      for (size_t i = 0; i < sampleCount; ++i) {
        data_t output(new std::vector<double>(visibleCount));
        thrust::copy(row(X, i).begin(), row(X, i).end(), output->begin());
        outputs->push_back(output);
      }
      newState->setOutputs(outputs);
      return;
    }

    matrix_t H;

    // Calculate p(h | X, W) = sigm(XW + C)
    H = prod(X, W);
    if (getDoubleWeights()) {
      if (!getOnlyFilters())
        H = 2.0 * H + repeat(c, H.size() / c.size());
      else
        H = 2.0 * H;
    } else if (!getOnlyFilters()) {
      H = H + repeat(c, H.size() / c.size());
    }

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

    if (getDoubleWeights()) {
      if (!getOnlyFilters())
        V = 2.0 * V + repeat(b, V.size() / b.size());
      else
        V = 2.0 * V;
    } else if (!getOnlyFilters()) {
      V = V + repeat(b, V.size() / b.size());
    }

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

    if (!getOnlyFilters())
      V = V * repeat(stddev, V.size() / stddev.size()) + repeat(mean, V.size() / mean.size());
    boost::shared_ptr<std::vector<data_t> > outputs(new std::vector<data_t>());
    for (size_t i = 0; i < sampleCount; ++i) {
      data_t output(new std::vector<double>(visibleCount));
      thrust::copy(row(V, i).begin(), row(V, i).end(), output->begin());
      outputs->push_back(output);
    }
    newState->setOutputs(outputs);
  }
#endif
}

}

}
