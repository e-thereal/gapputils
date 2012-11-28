/*
 * Trainer_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/dot.hpp>

#include <boost/timer.hpp>

namespace gml {

namespace convrbm {

TrainerChecker::TrainerChecker() {
  Trainer trainer;
  trainer.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRate, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);

  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(Filters, trainer);
}

#define START size_t timerCycles = getEpochCount(); \
    boost::timer timer;

#define STOP cudaThreadSynchronize(); \
    std::cout << __LINE__ << ": " << timer.elapsed() << std::endl; \
    timer.restart();

#define TIMER_LOOP for(size_t iCycle = 0; iCycle < timerCycles; ++iCycle)

unsigned int upper_power_of_two(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace thrust::placeholders;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  const unsigned dimCount = Model::dimCount;
  typedef complex<value_t> complex_t;
  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor_t::dim_t dim_t;

  boost::shared_ptr<Model> crbm = getInitialModel()->clone();

  const size_t sampleCount = getTensors()->size();
  const size_t batchSize = getBatchSize();
  const size_t batchCount = sampleCount / batchSize;
  const size_t filterCount = crbm->getFilters()->size();
  const size_t epochCount = getEpochCount();

  std::vector<boost::shared_ptr<host_tensor_t> >& tensors = *getTensors();
  std::vector<boost::shared_ptr<host_tensor_t> > X;


  for (size_t i = 0; i < tensors.size(); ++i) {
    X.push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(*tensors[i])));
  }

  if (crbm->getVisibleUnitType() == UnitType::Gaussian) {
    dlog() << "Mean and sd = " << crbm->getMean() << ", " << crbm->getStddev();
    for (size_t i = 0; i < X.size(); ++i)
      *X[i] = (*X[i] - crbm->getMean()) / crbm->getStddev();
  }

  // Copy filters to the device
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<tensor_t > F, Finc;
  for (size_t i = 0; i < filters.size(); ++i) {
    tensor_t filter(filters[i]->size());
    thrust::copy(filters[i]->begin(), filters[i]->end(), filter.begin());
    F.push_back(filter);
    Finc.push_back(tensor_t(filter.size()));
  }

  value_t b = crbm->getVisibleBias(), binc = 0;
  std::vector<value_t>& c = *crbm->getHiddenBiases(), cinc(filterCount, 0);

  dlog() << "Trainer initialized.";

  dim_t size = X[0]->size(),
      filterSize = F[0].size(),
      hiddenSize = abs(size - filterSize) + 1,
      paddedSize = size + filterSize - 1,
      start = filterSize - 1;

  size_t layerVoxelCount = 1;
  for (size_t i = 0; i < dimCount; ++i)
    layerVoxelCount *= hiddenSize[i];

  value_t epsilonw =  getLearningRate() / batchSize / layerVoxelCount;      // Learning rate for weights
  value_t epsilonvb = getLearningRate() / batchSize / layerVoxelCount;      // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize / layerVoxelCount;      // Learning rate for biases of hidden units
  value_t epsilonsb = getSparsityWeight() / batchSize;                      // Sparsity weight
  value_t weightcost = 0.0002 * getLearningRate();
  value_t initialmomentum = 0.5; //65; // 0.5f;
  value_t finalmomentum = 0.9; // 65; // 0.9f;
  value_t momentum;

  dlog() << "Starting training";


  tensor_t v(size), vneg(size), vtemp(size), h(hiddenSize), Ftemp(filterSize), padded = zeros<value_t>(paddedSize);
  random_tensor<value_t, dimCount, true, uniform<value_t> > randu(hiddenSize);

  boost::timer timer;
  size_t timerCycles = getEpochCount();

  dim_t pospaddedSize = size, negpaddedSize = paddedSize;
  for (size_t i = 0; i < dimCount - 1; ++i) {
    pospaddedSize[i] = upper_power_of_two(size[i]);
    negpaddedSize[i] = upper_power_of_two(paddedSize[i]);
  }
  dim_t fullHiddenSize = hiddenSize;
  fullHiddenSize[dimCount - 1] = size[dimCount - 1];

  plan_t plan, plan2, iplan, iplan2, negplan, negiplan;
  tensor_t padded1, padded2, negpadded1, negpadded2, fullH(fullHiddenSize), h2(hiddenSize), h3(hiddenSize), paddedH;
  ctensor_t ctens1, ctens2, ctens3, negctens1, negctens2;

  for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
    value_t error = 0;
    momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

    for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {

      for (size_t k = 0; k < filterCount; ++k) {
        Finc[k] = momentum * Finc[k] - weightcost * F[k];
        cinc[k] = momentum * cinc[k];
      }
      binc = momentum * binc;

      for (size_t iSample = 0; iSample < batchSize; ++iSample) {

        // get V
        thrust::copy(X[iSample + iBatch * batchSize]->begin(), X[iSample + iBatch * batchSize]->end(), v.begin());
        vneg = zeros<value_t>(size);

        for (size_t k = 0; k < filterCount; ++k) {

          /*** BEGIN OF POSITIVE PHASE ***/

          // h = conv(flip(F[k]), v);

          padded1 = zeros<value_t>(pospaddedSize);
          padded1[sequence<int,dimCount>(0), filterSize] = flip(flip(F[k]), dimCount - 1);
          ctens1 = fft(padded1, dimCount - 1, plan);

          if (k == 0) {
            padded2 = zeros<value_t>(pospaddedSize);
            padded2[sequence<int,dimCount>(0), size] = v;
            ctens2 = fft(padded2, dimCount - 1, plan);
          }

          ctens1 = ctens1 * ctens2;
          ctens3 = sum(ctens1, dimCount - 1);
          paddedH = ifft(ctens3, dimCount - 1, iplan);
          h3 = paddedH[size - fullHiddenSize, hiddenSize];

          h = sigm(h + c[k]);
//          Ftemp = conv(flip(h), v);

          padded1 = zeros<value_t>(pospaddedSize);
          padded1[sequence<int,dimCount>(0), hiddenSize] = flip(h);
          ctens1 = fft(padded1, plan2);
//          padded2 = zeros<value_t>(pospaddedSize);
//          padded2[sequence<int,dimCount>(0), size] = v;
//          ctens2 = fft(padded2, plan);
          ctens1 = ctens1 * ctens2;
          padded1 = ifft(ctens1, iplan2);
          Ftemp = padded1[size - filterSize, filterSize];

          Finc[k] = Finc[k] + epsilonw * Ftemp;
          cinc[k] = cinc[k] + epsilonhb * sum(h) + epsilonsb * (getSparsityTarget() - sum(h) / h.count());

          // Sample hidden state
          h = h > randu;

          /*** BEGIN OF NEGATIVE PHASE ***/

          padded[start, hiddenSize] = h;

//          vtemp = conv(F[k], padded);
          negpadded1 = zeros<value_t>(negpaddedSize);
          negpadded2 = zeros<value_t>(negpaddedSize);
          negpadded1[sequence<int,dimCount>(0), filterSize] = F[k];
          negpadded2[sequence<int,dimCount>(0), paddedSize] = padded;
          negctens1 = fft(negpadded1, negplan);
          negctens2 = fft(negpadded2, negplan);
          negctens1 = negctens1 * negctens2;
          negpadded1 = ifft(negctens1, negiplan);
          vtemp = negpadded1[paddedSize - size, size];

          vneg = vneg + vtemp;
        }

        /*** END OF POSITIVE PHASE ***/

        binc = binc + epsilonvb * sum(v);

        /*** END OF NEGATIVE PHASE ***/

        switch(crbm->getVisibleUnitType()) {
          case UnitType::Bernoulli: vneg = sigm(vneg + b); break;
          case UnitType::Gaussian:  vneg = vneg + b;       break;
          default:
            dlog(Severity::Warning) << "Unsupported unit type: " << crbm->getVisibleUnitType();
        }

        error += dot(vneg - v, vneg - v);

        for (size_t k = 0; k < filterCount; ++k) {
//          h = conv(flip(F[k]), vneg);
          padded1 = zeros<value_t>(pospaddedSize);
          padded1[sequence<int,dimCount>(0), filterSize] = flip(F[k]);
          ctens1 = fft(padded1, plan);
          if (k == 0) {
            padded2 = zeros<value_t>(pospaddedSize);
            padded2[sequence<int,dimCount>(0), size] = vneg;
            ctens2 = fft(padded2, plan);
          }
          ctens1 = ctens1 * ctens2;
          padded1 = ifft(ctens1, iplan);
          h = padded1[size - hiddenSize, hiddenSize];

          h = sigm(h + c[k]);

//          Ftemp = conv(flip(h), vneg);
          padded1 = zeros<value_t>(pospaddedSize);
          padded1[sequence<int,dimCount>(0), hiddenSize] = flip(h);
          ctens1 = fft(padded1, plan);
//          padded2 = zeros<value_t>(pospaddedSize);
//          padded2[sequence<int,dimCount>(0), size] = vneg;
//          ctens2 = fft(padded2, plan);
          ctens1 = ctens1 * ctens2;
          padded1 = ifft(ctens1, iplan);
          Ftemp = padded1[size - filterSize, filterSize];

          Finc[k] = Finc[k] - epsilonw * Ftemp;
          cinc[k] = cinc[k] - epsilonhb * sum(h);
        }

        binc = binc - epsilonvb * sum(vneg);
      } /* end of sample */

      for (size_t k = 0; k < filterCount; ++k) {
        F[k] = F[k] + Finc[k];
        c[k] = c[k] + cinc[k];
      }
      b += binc;

      if (monitor)
        monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    } /* end of batch */

    dlog(Severity::Trace) << "Error: " << error / sampleCount;

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount);
  }

  for (size_t i = 0; i < filterCount; ++i)
    thrust::copy(F[i].begin(), F[i].end(), filters[i]->begin());

  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
  for (size_t i = 0; i < F.size(); ++i)
    debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(F[i])));

  crbm->setVisibleBias(b);

  dlog() << "VisibleBias: " << b << " and " << crbm->getVisibleBias();

  newState->setFilters(debugFilters);
  newState->setModel(crbm);
}

}

}
