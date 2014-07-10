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

#include <tbblas/deeplearn/rbm.hpp>

#include <boost/timer.hpp>

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

  model_t& model = *getModel();
  tbblas::deeplearn::rbm<value_t> rbm(model);
  v_data_t& given = *getGiven();

  // Calculate the mean and the std of all features
  const size_t visibleCount = model.weights().size()[0];
  const size_t hiddenCount = model.weights().size()[1];
  const size_t givenCount = given[0]->size();
  const size_t sampleCount = given.size();
  const int firstGiven = getFirstGiven();

  if (firstGiven < 0 || givenCount + firstGiven >= visibleCount) {
    dlog(Severity::Warning) << "Given units exceed the number of the visible units. Aborting!";
    return;
  }

  rbm.visibles().resize(seq(sampleCount, visibleCount));

  matrix_t X(sampleCount, givenCount);    // all given units (used to reset the given units after each iteration)

  for (size_t i = 0; i < sampleCount; ++i)
    thrust::copy(given[i]->begin(), given[i]->end(), row(rbm.visibles(), i).begin() + firstGiven);

  // Normalize the given values
  rbm.normalize_visibles();
  X = rbm.visibles()[seq(0,firstGiven), X.size()];

  for (int i = 0; i < getIterationCount(); ++i) {
    rbm.visibles()[seq(0,firstGiven), X.size()] = X;      // Replace given units

    rbm.infer_hiddens();
    rbm.infer_visibles();

    if (monitor) {
      monitor->reportProgress(100.0 * (i + 1) / getIterationCount());
    }
  }

  rbm.diversify_visibles();

  boost::shared_ptr<v_data_t> outputs(new v_data_t());
  for (size_t i = 0; i < sampleCount; ++i) {
    boost::shared_ptr<data_t> output(new data_t(visibleCount));
    thrust::copy(row(rbm.visibles(), i).begin(), row(rbm.visibles(), i).end(), output->begin());
    outputs->push_back(output);
  }
  newState->setInferred(outputs);
}

}

}
