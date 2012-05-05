/*
 * ConvRbmTrainer_gpu.cu
 *
 *  Created on: Mar 5, 2012
 *      Author: tombr
 */

#define BOOST_TYPEOF_COMPLIANT
#include "ConvRbmTrainer.h"

#include <iostream>
#include <sstream>

#include <capputils/Verifier.h>

#include <boost/timer.hpp>
#include <tbblas/tensor_proxy.hpp>

//#include "sampling.hpp"

#include "RbmModel.h"

#include <curand.h>
#include <culib/CulibException.h>
#include <culib/util.h>

namespace gapputils {

namespace ml {

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl

template<class T>
struct softmax : thrust::binary_function<T, unsigned, T> {

  softmax(unsigned width, unsigned blockSize) : width(width), blockSize(blockSize) { }

  __host__ __device__
  T operator()(const T& value, const unsigned& idx) const {
    T res = 0;
    const int offset = (idx % width) % blockSize + ((idx / width) % blockSize) * width;
    for (unsigned j = 0; j < blockSize; ++j)
      for (unsigned i = 0; i < blockSize; ++i)
        res += exp(*(&value + i + j * width - offset));
    return exp(value) / (1 + res);
  }

private:
  unsigned blockSize, width;
};

void printMemoryAtLine(int line) {
  std::stringstream stream;
  stream << "line " << line;
  culib::printMemoryStats(stream.str().c_str());
  CULIB_CHECK_ERROR();
}

#define TRACE printMemoryAtLine(__LINE__);
//#define TRACE


void ConvRbmTrainer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace thrust::placeholders;

  boost::timer timer;

  typedef tbblas::tensor_proxy<device_tensor_t::iterator, 3> device_proxy_t;

  if (!data)
    data = new ConvRbmTrainer();

  //  std::cout << "Device:" << std::endl;
  //  ConvRbmTrainer test;
  //  LOCATE(test, InitialModel);
  //  LOCATE(test, Tensors);
  //  LOCATE(test, SampleVisibles);
  //  LOCATE(test, EpochCount);
  //  LOCATE(test, BatchSize);
  //  LOCATE(test, LearningRate);
  //  LOCATE(test, Model);

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInitialModel()) {
    std::cout << "[Warning] No initial model given. Aborting!" << std::endl;
    return;
  }

  if (!getTensors() || getTensors()->size() == 0) {
    std::cout << "[Warning] No training data given. Aborting!" << std::endl;
    return;
  }

  std::cout << "Building ConvRBM ..." << std::endl;

  TRACE

  curandGenerator_t gen;
  curandStatus_t status;
  if ((status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)) != CURAND_STATUS_SUCCESS) {
    std::cout << "[Warning] Could not create random number generator: " << status << std::endl;
    return;
  }

  const unsigned sampleCount = getTensors()->size();
  const int batchSize = getBatchSize();

  boost::shared_ptr<ConvRbmModel> crbm = getInitialModel()->clone();

  const unsigned dimCount = ConvRbmModel::dimCount;
  const unsigned filterCount = crbm->getFilters()->size();
  const unsigned blockSize = crbm->getPoolingBlockSize();
  const host_tensor_t::dim_t& filterDim = crbm->getFilters()->at(0)->size();
  const host_tensor_t::dim_t& inputDim = getTensors()->at(0)->size();
  host_tensor_t::dim_t layerDim, paddedDim, start;

  int filterWeightCount = 1, layerVoxelCount = 1, inputVoxelCount = 1;
  for (unsigned i = 0; i < dimCount; ++i) {
    layerDim[i] = inputDim[i] - filterDim[i] + 1;
    paddedDim[i] = inputDim[i] + filterDim[i] - 1;
    start[i] = filterDim[i] - 1;
    filterWeightCount *= filterDim[i];
    layerVoxelCount *= layerDim[i];
    inputVoxelCount *= inputDim[i];
  }

  assert((layerDim[0] % blockSize) == 0);
  assert((layerDim[1] % blockSize) == 0);
  assert((layerVoxelCount % 2) == 0); // TODO: loosen this constrain. Means, use temporary array to generate
  assert((inputVoxelCount % 2) == 0); //       random number (count must be a multiple of 2)

  // Train the RBM
  std::vector<boost::shared_ptr<host_tensor_t> >& tensors = *getTensors();
  std::vector<boost::shared_ptr<host_tensor_t> > X;

  for (unsigned i = 0; i < tensors.size(); ++i) {
    X.push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(tbblas::copy(*tensors[i]))));
  }

  if (crbm->getIsGaussian()) {
    // Calculate the mean and normalize the data
    value_t mean = crbm->getMean();
    value_t stddev = crbm->getStddev();

    for (unsigned i = 0; i < X.size(); ++i)
      *X[i] += -mean;

    for (unsigned  i = 0; i < X.size(); ++i) {
      *X[i] = tbblas::copy(*X[i] / stddev);
    }
  }

  // Copy filters to the device
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<device_tensor_t > F;
  for (unsigned i = 0; i < filters.size(); ++i) {
    device_tensor_t filter(filters[i]->size());
    thrust::copy(filters[i]->begin(), filters[i]->end(), filter.begin());
    F.push_back(filter);
  }

  value_t b = crbm->getVisibleBias();
  std::vector<value_t>& c = *crbm->getHiddenBiases();

  std::cout << "[Info] ConvRBM initialized: " << timer.elapsed() << " s" << std::endl;

  // Start the learning
  const int batchCount = sampleCount / batchSize;
  value_t epsilonw =  getLearningRate();      // Learning rate for weights
  value_t epsilonvb = getLearningRate();      // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate();      // Learning rate for biases of hidden units
  value_t weightcost = 0; // 0.0002;
  value_t initialmomentum = 0.5; //65; // 0.5f;
  value_t finalmomentum = 0.9; // 65; // 0.9f;
  value_t momentum;

  culib::printMemoryStats("ConvRbmTrainer initialized");

  device_tensor_t v(inputDim), vneg(inputDim), vtemp(inputDim), padded(paddedDim);
  thrust::fill(padded.begin(), padded.end(), value_t(0));
  std::vector<device_tensor_t> poshidprobs, poshidstates, posvishid, neghidprobs, neghidstates, negvishid, Finc, Fincbatch;
  for (unsigned i = 0; i < filterCount; ++i) {
    poshidprobs.push_back(device_tensor_t(layerDim));
    poshidstates.push_back(device_tensor_t(layerDim));
    posvishid.push_back(device_tensor_t(filterDim));

    neghidprobs.push_back(device_tensor_t(layerDim));
    neghidstates.push_back(device_tensor_t(layerDim));
    negvishid.push_back(device_tensor_t(filterDim));
    Finc.push_back(device_tensor_t(filterDim));
    thrust::fill(Finc[i].begin(), Finc[i].end(), value_t(0));
    Fincbatch.push_back(device_tensor_t(filterDim));
  }
  value_t posvisact, negvisact, binc = 0, bincbatch;
  std::vector<value_t> poshidact(filterCount), neghidact(filterCount),
      cinc(filterCount, 0), cincbatch(filterCount, 0),
      cspa(filterCount, 0), cspabatch(filterCount, 0);

  const int epochCount = getEpochCount();

  std::cout << "[Info] Preparation finished after " << timer.elapsed() << " s" << std::endl;
  CULIB_CHECK_ERROR();
  culib::printMemoryStats("ConvRbmTrainer memory allocated");
  std::cout << "[Info] Starting training" << std::endl;
  timer.restart();

  if (epochCount && getShowProgress()) {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
        new std::vector<boost::shared_ptr<host_tensor_t> >());
    for (unsigned i = 0; i < filterCount; ++i)
      debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(tbblas::copy(F[i]))));
    data->setFilters(debugFilters);
  }

  if (monitor)
    monitor->reportProgress(0, getShowProgress());

  for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    double error = 0;
    for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

      for (unsigned k = 0; k < filterCount; ++k) {
        thrust::fill(Fincbatch[k].begin(), Fincbatch[k].end(), value_t(0));
        cincbatch[k] = 0;
        cspabatch[k] = 0;
      }
      bincbatch = 0;

      for (int iSample = 0; iSample < batchSize && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {

        /*** START POSITIVE PHASE ***/
        const int randomSample = rand() % sampleCount;

        // Get current sample
        if (getUseRandomSamples())
          thrust::copy(X[randomSample]->begin(), X[randomSample]->end(), v.begin());
        else
          thrust::copy(X[iSample + iBatch * batchSize]->begin(), X[iSample + iBatch * batchSize]->end(), v.begin());

        // For each filter (Could be written as a single 4D convolution in case of a 2D image and 3D filter))
        for (unsigned k = 0; k < filterCount && (monitor ? !monitor->getAbortRequested() : true); ++k) {

          // Calculate p(h_k | v, F) = sigm((~F_k * v) + c_k)
          poshidstates[k] = tbblas::conv(tbblas::flip(F[k]), v, (k ? tbblas::ReuseFT2 : tbblas::ReuseFTNone));
          poshidstates[k] += c[k];               // x = ~F_k * v + c_k

          // I'm using the state array here for the sum. Not nice but works fine and saves some space
          thrust::transform(poshidstates[k].data().begin(), poshidstates[k].data().end(),
              thrust::make_counting_iterator(0), poshidprobs[k].data().begin(),
              softmax<value_t>(layerDim[0], blockSize));

//          thrust::transform(poshidprobs[k].data().begin(), poshidprobs[k].data().end(), // x = sigm(x)
//              poshidprobs[k].data().begin(), sigmoid<value_t>());

          // Calculate energy and the total activation of the hidden units
          posvishid[k] = tbblas::conv(tbblas::flip(poshidprobs[k]), v, tbblas::ReuseFT2);     // ~h_k * v
          poshidact[k] = tbblas::sum(poshidprobs[k]);

          if (iEpoch || !getCalculateBaseline())
            cspabatch[k] += getSparsityTarget() - tbblas::sum(poshidprobs[k]) / poshidprobs[k].data().size();

          // fill states with random numbers which are then used to sample the units
          // TODO: use curandGenerateUniform if value_t == float
          if ((status = curandGenerateUniformDouble(gen,
              poshidstates[k].data().data().get(),
              poshidstates[k].data().size())) != CURAND_STATUS_SUCCESS)
          {
            std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
            return;
          }

          // Sample the hidden states
          // TODO: sample correctly from the categorical.
          thrust::transform(
              poshidprobs[k].data().begin(), poshidprobs[k].data().end(), poshidstates[k].data().begin(),
              poshidstates[k].data().begin(), _1 > _2
          );
        }

        // Calculate the total activation of the visible units
        posvisact = tbblas::sum(v);

        /*** END OF POSITIVE PHASE ***/

        /*** START NEGATIVE PHASE ***/

        // Calculate p(v | H, F) = sigm(sum(W_k * h_k) + b)
        thrust::fill(vneg.data().begin(), vneg.data().end(), value_t(0));
        for (unsigned k = 0; k < filterCount; ++k) {
          device_proxy_t paddedProxy = tbblas::subrange(padded, start, layerDim);
          thrust::copy(poshidstates[k].begin(), poshidstates[k].end(), paddedProxy.begin());
          vtemp = tbblas::conv(F[k], padded);
          vneg += vtemp;
        }
        vneg += b;

        // For the binary case
        if (!crbm->getIsGaussian()) {
          thrust::transform(vneg.begin(), vneg.end(), vneg.begin(),
              sigmoid<value_t>());

          if (getSampleVisibles()) {
            if ((status = curandGenerateUniformDouble(gen, vtemp.data().data().get(), vtemp.data().size())) != CURAND_STATUS_SUCCESS)
            {
              std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
              return;
            }

            thrust::transform(
                vneg.data().begin(), vneg.data().end(), vtemp.data().begin(),
                vneg.data().begin(), _1 > _2
            );
          }
        } else {
          if (getSampleVisibles()) {
            if ((status = curandGenerateNormalDouble(gen,
                vtemp.data().data().get(),
                vtemp.data().size(),
                0, 1.0)) != CURAND_STATUS_SUCCESS)
            {
              std::cout << "[Error] Could not generate random numbers: " << status << std::endl;
              return;
            }

            thrust::transform(
                vneg.data().begin(), vneg.data().end(), vtemp.data().begin(),
                vneg.data().begin(), thrust::plus<value_t>()
            );
          }
        }

        for (unsigned k = 0; k < filterCount; ++k) {

          // Calculate p(h_k | vneg, F) = sigm((~F_k * v) + c_k)
          neghidstates[k] = tbblas::conv(tbblas::flip(F[k]), vneg,
              (k ? tbblas::ReuseFT2 : tbblas::ReuseFTNone));               // x = ~F_k * v + c_k
          neghidstates[k] += c[k];

          thrust::transform(neghidstates[k].data().begin(), neghidstates[k].data().end(),
              thrust::make_counting_iterator(0), neghidprobs[k].data().begin(),
              softmax<value_t>(layerDim[0], blockSize));

//          thrust::transform(neghidprobs[k].data().begin(), neghidprobs[k].data().end(), // x = sigm(x)
//              neghidprobs[k].data().begin(), sigmoid<value_t>());

          // Calculate energy and the total activation of the hidden units
          negvishid[k] = tbblas::conv(tbblas::flip(neghidprobs[k]), vneg, tbblas::ReuseFT2);     // ~h_k * v
          neghidact[k] = tbblas::sum(neghidprobs[k]);
        }

        // Calculate the total activation of the visible units
        negvisact = tbblas::sum(vneg);

        /*** END OF NEGATIVE PHASE ***/

        double curerr = thrust::inner_product(vneg.begin(), vneg.end(), v.begin(), value_t(0),
            thrust::plus<value_t>(), (_1 - _2) * (_1 - _2));
        error += curerr;
        momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

        /*** UPDATE WEIGHTS AND BIASES ***/
        if (iEpoch || !getCalculateBaseline()) {
          for (unsigned k = 0; k < filterCount; ++k) {
            Fincbatch[k] += (posvishid[k] += (-1.0 * negvishid[k]));
            cincbatch[k] += (poshidact[k] - neghidact[k]);
          }
          bincbatch = posvisact - negvisact;
        }
      }
      for (unsigned k = 0; k < filterCount; ++k) {
        Finc[k] = momentum * Finc[k] + (epsilonw / batchSize / layerVoxelCount) * Fincbatch[k];
        cinc[k] = momentum * cinc[k] + (epsilonhb / batchSize / layerVoxelCount) * cincbatch[k]
                  + getSparsityPenalty() * cspabatch[k] / batchSize;

        F[k] += Finc[k];
        c[k] += cinc[k];
      }
      binc = momentum * binc + (epsilonvb / batchSize / inputVoxelCount) * bincbatch;
      b += binc;

      /*** END OF UPDATES ***/

      if (monitor)
        monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    }
    int eta = (int)(timer.elapsed() / (double)(iEpoch + 1) * (double)(epochCount - iEpoch - 1));
    int sec = eta % 60;
    int minutes = (eta / 60) % 60;
    int hours = eta / 3600;
    std::cout << "Epoch " << iEpoch << " error " << (error / sampleCount) << " after " << timer.elapsed() << "s. ETA: "
        << hours << " h " << minutes << " min " << sec << " s" << std::endl;

    if (getShowProgress()){
      boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
          new std::vector<boost::shared_ptr<host_tensor_t> >());
      for (unsigned i = 0; i < filterCount; ++i)
        debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(tbblas::copy(F[i]))));
      data->setFilters(debugFilters);
    }

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount, (iEpoch < epochCount - 1) && getShowProgress());
  }

  if ((status = curandDestroyGenerator(gen)) != CURAND_STATUS_SUCCESS)
  {
    std::cout << "[Error] Could not destroy random number generator: " << status << std::endl;
    return;
  }

  {
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
    for (unsigned i = 0; i < filterCount; ++i)
      debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(tbblas::copy(F[i]))));
    data->setFilters(debugFilters);
  }

  for (unsigned i = 0; i < filterCount; ++i) {
    thrust::copy(F[i].begin(), F[i].end(), filters[i]->begin());
  }
  data->setModel(crbm);
}

}

}


