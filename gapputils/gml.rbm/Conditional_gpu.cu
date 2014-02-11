/*
 * Conditional_gpu.cu
 *
 *  Created on: Nov 15, 2013
 *      Author: tombr
 */

#include "Conditional.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>
#include <tbblas/linalg.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/io.hpp>

#include <boost/timer.hpp>

#include "math.hpp"

namespace gml {

namespace rbm {

ConditionalChecker::ConditionalChecker() {
  Conditional test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Given, test);
  CHECK_MEMORY_LAYOUT2(FirstGiven, test);
  CHECK_MEMORY_LAYOUT2(IterationCount, test);
  CHECK_MEMORY_LAYOUT2(Inferred, test);
}

void Conditional::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  // Replace the given units
  // Mean field approximation
  // Infer the hiddens
  // Infer the visibles

  using namespace tbblas;

//  typedef random_tensor<value_t, 2, true, normal<value_t> > randn_t;
//  typedef random_tensor<value_t, 2, true, uniform<value_t> > randu_t;

  Model& rbm = *getModel();
  v_data_t& given = *getGiven();

  UnitType visibleUnitType = rbm.getVisibleUnitType();
  UnitType hiddenUnitType = rbm.getHiddenUnitType();

  matrix_t W = *rbm.getWeightMatrix();
  matrix_t b = *rbm.getVisibleBiases();
  matrix_t c = *rbm.getHiddenBiases();
  matrix_t mean =  *rbm.getMean();
  matrix_t stddev = *rbm.getStddev();

  // Calculate the mean and the std of all features
  const size_t visibleCount = W.size()[0];
  const size_t hiddenCount = W.size()[1];
  const size_t givenCount = given[0]->size();
  const size_t sampleCount = given.size();
  const int firstGiven = getFirstGiven();

  if (firstGiven < 0 || givenCount + firstGiven >= visibleCount) {
    dlog(Severity::Warning) << "Given units exceed the number of the visible units. Aborting!";
    return;
  }

  matrix_t V(sampleCount, visibleCount);  // all visible units
  matrix_t H(sampleCount, hiddenCount);   // all hidden units
  matrix_t X(sampleCount, givenCount);    // all given units

  for (size_t i = 0; i < sampleCount; ++i)
    thrust::copy(given[i]->begin(), given[i]->end(), row(V, i).begin() + firstGiven);

  // Normalize the given values
  V = (V - repeat(mean, V.size() / mean.size())) / repeat(stddev, V.size() / stddev.size());
  X = V[seq(0,0), X.size()];

  for (int i = 0; i < getIterationCount(); ++i) {
    V[seq(0,firstGiven), X.size()] = X;      // Replace given units

    // Calculate p(h | V, W) = sigm(VW + C)
    H = prod(V, W);
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

    // Calculate p(v | H, W) = sigm(HW' + B)
    V = prod(H, tbblas::trans(W));
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

    if (monitor) {
      monitor->reportProgress(100.0 * (i + 1) / getIterationCount());
    }
  }

  V = V * repeat(stddev, V.size() / stddev.size()) + repeat(mean, V.size() / mean.size());

  boost::shared_ptr<v_data_t> outputs(new v_data_t());
  for (size_t i = 0; i < sampleCount; ++i) {
    boost::shared_ptr<data_t> output(new data_t(visibleCount));
    thrust::copy(row(V, i).begin(), row(V, i).end(), output->begin());
    outputs->push_back(output);
  }
  newState->setInferred(outputs);
}

}

}
