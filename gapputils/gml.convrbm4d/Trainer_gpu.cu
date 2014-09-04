/*
 * Trainer_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

//#define TBBLAS_INTERRUPT_ALLOC_ENABLED
//#define TBBLAS_ALLOC_WARNING_ENABLED

#include "Trainer.h"

#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/io.hpp>

#include <boost/timer.hpp>

#include <fstream>

#include <tbblas/deeplearn/conv_rbm.hpp>

namespace gml {

namespace convrbm4d {

TrainerChecker::TrainerChecker() {
  Trainer trainer;
  trainer.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(DbmLayer, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);

  CHECK_MEMORY_LAYOUT2(SparsityMethod, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);

  CHECK_MEMORY_LAYOUT2(CdIterations, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRate, trainer);
  CHECK_MEMORY_LAYOUT2(LearningDecay, trainer);
  CHECK_MEMORY_LAYOUT2(InitialMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(FinalMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(MomentumDecayEpochs, trainer);
  CHECK_MEMORY_LAYOUT2(WeightDecay, trainer);
  CHECK_MEMORY_LAYOUT2(WeightVectorLimit, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
  CHECK_MEMORY_LAYOUT2(DropoutMethod, trainer);
  CHECK_MEMORY_LAYOUT2(VisibleDropout, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenDropout, trainer);
  CHECK_MEMORY_LAYOUT2(FilterDropout, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(UpdateModel, trainer);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, trainer);
  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(AverageEpochTime, trainer);
  CHECK_MEMORY_LAYOUT2(ReconstructionError, trainer);
}

#define START size_t timerCycles = getEpochCount(); \
    boost::timer _timer;

#define STOP { \
    cudaStreamSynchronize(0); \
    std::cout << __LINE__ << ": " << _timer.elapsed() << std::endl; \
    _timer.restart(); \
}

#define TIMER_LOOP for(size_t iCycle = 0; iCycle < timerCycles; ++iCycle)

#define TRACE std::cout << __LINE__ << std::endl;

#define DROPOUT

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef float value_t;
//
  const unsigned dimCount = model_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

#if 1
  /*** SETUP TRAINING PARAMETERS ***/

  const size_t batchSize = getBatchSize();
  const size_t filterBatchLength = getFilterBatchSize();
  const size_t batchCount = getTensors()->size() / batchSize;
  const size_t epochCount = getEpochCount();

  std::vector<boost::shared_ptr<host_tensor_t> >& X = *getTensors();

  if (filterBatchLength > getInitialModel()->filters().size() ||
      getInitialModel()->filters().size() % filterBatchLength != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  /*** PREPARE MASTER THREAD ***/

  // Make copying of model optional
#if 0
  conv_rbm<float, 4> newModel(getGpuCount());
  conv_rbm<float, 4>& crbm = oldModel();

  if (!getAtomicWorkflow() || oldCrbm.use_count() != 2) {
    // Copy model
    crbm = newModel;
  }
  crbm.set_batch_length(getFilterBatchSize());
#else
  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
  model->set_shared_bias(getShareBiasTerms());

  tbblas::deeplearn::conv_rbm<float, 4> crbm(*model, getGpuCount());
  crbm.set_batch_length(getFilterBatchSize());
  crbm.set_sparsity_method(getSparsityMethod());
  crbm.set_sparsity_target(getSparsityTarget());
  crbm.set_sparsity_weight(getSparsityWeight());
#endif

  // Prepare sizes
  size_t voxelCount = sum(model->mask()) * X[0]->size()[dimCount - 1];

  // Initialize constants
  value_t epsilonw =  getLearningRate() / batchSize;  // Learning rate for weights
  value_t epsilonvb = getLearningRate() / batchSize;  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize;  // Learning rate for biases of hidden units
  value_t weightcost = getWeightDecay() * getLearningRate();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  value_t error = 0;
  tensor_t v;

  /*** START OF PARALLEL CODE ***/

  crbm.allocate_gpu_memory();

  dlog() << "Trainer initialized. Starting training.";

  for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
    error = 0;

    if (iEpoch < getMomentumDecayEpochs()) {
      const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
      momentum = (1 - t) * initialmomentum + t * finalmomentum;
    } else {
      momentum = finalmomentum;
    }

    for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {

      // Apply momentum for next batch
      crbm.init_gradient_updates(momentum, weightcost);

      for (size_t iSample = 0; iSample < batchSize; ++iSample) {
        crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

        // Get new sample
        if (getRandomizeTraining())
          crbm.visibles() = *X[rand() % X.size()];
        else
          crbm.visibles() = *X[iSample + iBatch * batchSize];
        crbm.normalize_visibles();

        if (getCalculateError())
          v = crbm.visibles();

        for (size_t iCd = 0; iCd <= getCdIterations(); ++iCd) {

          /*** BEGIN OF POSITIVE PHASE ***/
          crbm.infer_hiddens();

          if (iCd == 0) {
            crbm.update_positive_gradient(epsilonw, epsilonvb, epsilonhb);
          } else if (iCd == getCdIterations()) {
            crbm.update_negative_gradient(epsilonw, epsilonvb, epsilonhb);

            if (getCalculateError()) {
              error += sqrt(dot((crbm.visibles() - v), (crbm.visibles() - v)) / voxelCount);
            }
          }

          /*** RECONSTRUCT FROM SAMPLES ***/

          if (iCd < getCdIterations()) {
            crbm.sample_hiddens();
            crbm.sample_visibles();
          }
        } /* end of cd iterations */
      } /* end of sample */

      crbm.apply_gradient();

      if (monitor)
        monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    } /* end of batch */

    epsilonw *= getLearningDecay();
    epsilonvb *= getLearningDecay();
    epsilonhb *= getLearningDecay();

    if (getCalculateError())
      dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / X.size();
    else
      dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount, getUpdateModel() && (iEpoch % getUpdateModel() == 0));
  } /* end of epochs */

//    newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
  newState->setReconstructionError(error / X.size());
  newState->setModel(model);

#else
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  const int gpuCount = getGpuCount();

//  if (deviceCount < gpuCount) {
//    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << gpuCount << " are required according to GpuCount. Aborting!";
//    return;
//  }

  assert(omp_get_num_threads() == 1);

  cudaSetDevice(0);
  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

  /*** SETUP TRAINING PARAMETERS ***/

  const size_t batchSize = getBatchSize();
  const size_t filterBatchLength = getFilterBatchSize();
  const size_t batchCount = getTensors()->size() / batchSize;
  const size_t epochCount = getEpochCount();

  if (filterBatchLength > getInitialModel()->getFilters()->size() ||
      getInitialModel()->getFilters()->size() % filterBatchLength != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  // Prepare sizes
  dim_t originalSize = getTensors()->at(0)->size();
  dim_t size = originalSize;

//  if (getPadInputs()) {
//    for (unsigned j = 0; j < dimCount - 1; ++j) {
//      size[j] = upper_power_of_two(size[j]);
//    }
//  }

  dim_t originalLayerSize = originalSize, layerSize = size, layerBatchSize = size, filterBatchSize = size;
  originalLayerSize[dimCount - 1] = layerSize[dimCount - 1] = 1;
  filterBatchSize[dimCount - 1] = size[dimCount - 1] * filterBatchLength;
  layerBatchSize[dimCount - 1] = filterBatchLength;

  // Test if the FFT bug will bug us ;)
  {
    random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(size);

    tensor_t A = v_noise, B = A;
    ctensor_t cA = fft(A, 3), cB = cA;

    if (dot(A - B, A - B) > 0) {
      dlog(Severity::Warning) << "Bug in cuFFT detected. FFT changes its input. Padding of inputs to a power of 2 might help. Aborting!";
      return;
    }

    A = ifft(cA, 3);

    if (abs(dot(cA - cB, cA - cB)) > 0) {
      dlog(Severity::Warning) << "Bug in cuFFT detected. FFT changes its input. Padding of inputs to a power of 2 might help. Aborting!";
      return;
    }
  }

  {
    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise(layerBatchSize);

    tensor_t A = h_noise, B = A;
    ctensor_t cA = fft(A, 3), cB = cA;

    if (dot(A - B, A - B) > 0) {
      dlog(Severity::Warning) << "Bug in cuFFT detected. FFT changes its input. Padding of inputs to a power of 2 might help. Aborting!";
      return;
    }

    A = ifft(cA, 3);

    if (abs(dot(cA - cB, cA - cB)) > 0) {
      dlog(Severity::Warning) << "Bug in cuFFT detected. FFT changes its input. Padding of inputs to a power of 2 might help. Aborting!";
      return;
    }
  }

  size_t voxelCount = sum(*getInitialModel()->getMask()) * size[dimCount - 1];

  // Initialize constants
  value_t epsilonw =  getLearningRate() / batchSize / voxelCount; // Learning rate for weights
  value_t epsilonsw = getLearningRate() * getSparsityWeight() / batchSize / voxelCount; // Sparsity weight
  value_t epsilonvb = getLearningRate() / batchSize / size[dimCount - 1];                  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize / size[dimCount - 1];                  // Learning rate for biases of hidden units
  value_t epsilonsb = getLearningRate() * getSparsityWeight() / batchSize / size[dimCount - 1];                  // Sparsity weight
  value_t weightcost = getWeightDecay() * getLearningRate() / size[dimCount - 1];
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  /*** PREPARE MASTER THREAD ***/

  boost::shared_ptr<Model> crbm(new Model());
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > filters;
  boost::shared_ptr<host_tensor_t> b;
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > c;
  {
    // Copy the old model as far as necessary
    boost::shared_ptr<Model> oldCrbm = getInitialModel();
    filters = oldCrbm->getFilters();
    b = oldCrbm->getVisibleBias();
    c = oldCrbm->getHiddenBiases();
    crbm->setFilterKernelSize(oldCrbm->getFilterKernelSize());
    crbm->setMean(oldCrbm->getMean());
    crbm->setStddev(oldCrbm->getStddev());
    crbm->setVisibleUnitType(oldCrbm->getVisibleUnitType());
    crbm->setHiddenUnitType(oldCrbm->getHiddenUnitType());
    crbm->setConvolutionType(oldCrbm->getConvolutionType());
    crbm->setMask(boost::make_shared<host_tensor_t>(*oldCrbm->getMask()));

    // Prepare an early memory clean up of the old model
    // The trainer holds two pointers to the model, hence if the use_count is 2
    // no other modules have any business with the old model any more
    if (getAtomicWorkflow() && oldCrbm.use_count() == 2) {
      oldCrbm->setFilters(boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >());
      oldCrbm->setVisibleBias(boost::shared_ptr<host_tensor_t>());
      oldCrbm->setHiddenBiases(boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >());
    }
  }

#ifdef MONITOR_TRAINING

  boost::shared_ptr<Model> posInc(new Model());
  boost::shared_ptr<Model> inc(new Model());

  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > filters_inc(
      new std::vector<boost::shared_ptr<host_tensor_t> >(filters->size()));
  boost::shared_ptr<host_tensor_t> b_inc;
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > c_inc(
      new std::vector<boost::shared_ptr<host_tensor_t> >(c->size()));

  inc->setFilters(filters_inc);
  inc->setHiddenBiases(c_inc);

#endif

  // Normalize input and pre-calculate the FFT
  // In case of valid convolutions: pad the input
  std::vector<boost::shared_ptr<host_ctensor_t> > cX;
  {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > tensors = getTensors();
    tensor_t x, vMask;
    ctensor_t cx;
    plan_t plan_v;

    // Pad the mask
    vMask = zeros<value_t>(layerSize);
    vMask[seq(0,0,0,0), crbm->getMask()->size()] = *crbm->getMask();

    for (size_t i = 0; i < tensors->size(); ++i) {
      x = zeros<value_t>(size);
      x[seq(0,0,0,0), tensors->at(i)->size()] = *tensors->at(i);

      for (unsigned j = 0; j < dimCount - 1; ++j) {
        if (tensors->at(i)->size()[j] > size[j]) {
          dlog(Severity::Warning) << "Input tensors must have the same size. Aborting!";
          return;
        }
      }

      if (crbm->getVisibleUnitType() == UnitType::Gaussian)
        x = (x - crbm->getMean()) / crbm->getStddev();
      x = x * repeat(vMask, size / layerSize);
      cx = fft(x, dimCount - 1, plan_v);
      cX.push_back(boost::shared_ptr<host_ctensor_t>(new host_ctensor_t(cx)));

      if (getAtomicWorkflow() && tensors.use_count() == 2) {
        tensors->at(i) = boost::shared_ptr<host_tensor_t>();
      }
    }

    // Prepare an early memory clean up of the training set
    // The trainer holds two pointers to the training set, hence if the use_count is 2
    // no other modules have any business with the training set any more
    if (getAtomicWorkflow() && tensors.use_count() == 2)
      tensors->clear();
  }
  dlog() << "FFTs precalculated.";

  std::vector<boost::shared_ptr<ctensor_t> > cF_master(filters->size() / filterBatchLength);
  std::vector<boost::shared_ptr<ctensor_t> > cc_master(c->size() / filterBatchLength);

#ifdef DROPOUT
  std::vector<boost::shared_ptr<tensor_t> > drops(cF_master.size());
  std::vector<bool> dropFilter(cF_master.size());
#endif

  // Copy visible bias to the device
#ifdef SPREAD_IMAGES
  ctensor_t *cb_master, *cbinc_master;
#endif

  // Declare variables used for training
  ctensor_t* cv_master;      // Read from the master thread and then each other thread reads from the master device
  ctensor_t* cvneg_master;   // All threads add their version to the master threads version (needs to be synchronized)

  dim_t vbMaskSize, hbMaskSize, spMaskSize;

#ifdef DROPOUT
  tensor_t drop_master;
#endif

  value_t error = 0;

  /*** START OF PARALLEL CODE ***/

  #pragma omp parallel
  {
    /*** PREPARE GPU THREADS ***/

    int tid = omp_get_thread_num();
    cudaSetDevice(tid % deviceCount);

    // Enable peer to peer access of each card with the master card and vice versa
    if (tid < deviceCount) {
      if (tid == 0) {
        for (int i = 1; i < min(deviceCount, gpuCount); ++i)
          cudaDeviceEnablePeerAccess(i, 0);
      } else {
        cudaDeviceEnablePeerAccess(0, 0);
      }
    }
    #pragma omp barrier

    #pragma omp master
    assert(tid == 0);   // Check the assumption that the first thread is the master thread

    std::vector<boost::shared_ptr<ctensor_t> > cF(filters->size() / filterBatchLength), cFinc(cF.size());
    std::vector<boost::shared_ptr<ctensor_t> > cc(c->size() / filterBatchLength), ccinc(cc.size());

    tensor_t v, vneg;          // to monitor training (calculate the error)

    value_t batchError = 0;

    // FFT plans
    plan_t plan_h, iplan_h, plan_v, iplan_v;  // h is always a batch of hs

    // Copy visible bias to the device
    ctensor_t cb, cbinc;

#ifndef SPREAD_IMAGES
    #pragma omp master
#endif
    {
      tensor_t f = zeros<value_t>(size);
      f[seq(0,0,0,0), b->size()] = *b;
      plan_t plan_v;
      if (getShareBiasTerms())
        f = ones<value_t>(f.size()) * sum(f) / f.count();
      cb = fft(f, dimCount - 1, plan_v);
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
      b.reset();
    }

#ifdef SPREAD_IMAGES
    #pragma omp master
    {
      cb_master = &cb;
      cbinc_master = &cbinc;
    }
#endif

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, k, p;   // filter, kernel, padded kernel
      ctensor_t cf;
#ifdef SPREAD_IMAGES
      for (size_t i = 0; i < cF.size(); ++i) {
#else
      for (size_t i = tid; i < cF.size(); i += gpuCount) {
#endif

        p = zeros<value_t>(filterBatchSize);
        for (size_t j = 0; j < filterBatchLength; ++j) {
          k = *filters->at(i * filterBatchLength + j);
          dim_t topleft = size / 2 - k.size() / 2;
          topleft[dimCount - 1] = j * size[dimCount - 1];
          p[topleft, k.size()] = k;
        }
        f = ifftshift(p, dimCount - 1);

        cf = fft(f, dimCount - 1, plan_v);
        cF[i] = boost::make_shared<ctensor_t>(cf);
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cf.size(), cf.fullsize()));

#ifdef SPREAD_IMAGES
        #pragma omp master
        cF_master[i] = cF[i];
#endif
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier

      #pragma omp master
      {
        filters = boost::make_shared<std::vector<boost::shared_ptr<host_tensor_t> > >(filters->size());
        crbm->setFilters(filters);
      }
    }
    {
      tensor_t h, hBatch(layerBatchSize);
      ctensor_t ch;
#ifdef SPREAD_IMAGES
      for (size_t i = 0; i < cc.size(); ++i) {
#else
      for (size_t i = tid; i < cc.size(); i += gpuCount) {
#endif
        for (int j = 0; j < filterBatchLength; ++j) {
          h = *c->at(i * filterBatchLength + j);
          if (getShareBiasTerms())
            h = ones<value_t>(h.size()) * sum(h) / h.count();
          hBatch[seq(0,0,0,j), layerSize] = zeros<value_t>(layerSize);
          hBatch[seq(0,0,0,j), h.size()] = h;
        }
        ch = fft(hBatch, dimCount - 1, plan_h);
        cc[i] = boost::make_shared<ctensor_t>(ch);
        ch = zeros<complex_t>(ch.size(), ch.fullsize());
        ccinc[i] = boost::make_shared<ctensor_t>(ch);

#ifdef SPREAD_IMAGES
        #pragma omp master
        cc_master[i] = cc[i];
#endif

#ifdef DROPOUT
        drops[i] = boost::make_shared<tensor_t>();
#endif
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier

      #pragma omp master
      {
        c = boost::make_shared<std::vector<boost::shared_ptr<host_tensor_t> > >(c->size());
        crbm->setHiddenBiases(c);
      }
    }

    // Declare variables used for training
    tensor_t h, h2, f, vMask, hMask;        // for sigm, sampling, filter masking and visible masking
    ctensor_t ch, chdiff, ch_full;
    ctensor_t cv;                   // Read from the master thread and then each other thread reads from the master device
    ctensor_t cvneg;                // All threads add their version to the master threads version (needs to be synchronized)

    dim_t hsize = originalSize;
    hsize[dimCount - 1] = cF.size();

    #pragma omp master
    {
      cv_master = &cv;
      cvneg_master = &cvneg;
    }

    random_tensor<value_t, dimCount, true, uniform<value_t> > h_rand;
    random_tensor<value_t, dimCount, true, uniform<value_t> > v_rand;
    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise;
    random_tensor<value_t, dimCount, true, normal<value_t> > v_noise;

    if (getDropoutMethod() != DropoutMethod::NoDrop || crbm->getHiddenUnitType() == UnitType::Bernoulli)
      h_rand.resize(layerBatchSize, tid);

    if (crbm->getHiddenUnitType() == UnitType::MyReLU ||
        crbm->getHiddenUnitType() == UnitType::ReLU ||
        crbm->getHiddenUnitType() == UnitType::ReLU1 ||
        crbm->getHiddenUnitType() == UnitType::ReLU2 ||
        crbm->getHiddenUnitType() == UnitType::ReLU4)
    {
      h_noise.resize(layerBatchSize, tid);
    }

    if (crbm->getVisibleUnitType() == UnitType::MyReLU ||
        crbm->getVisibleUnitType() == UnitType::ReLU ||
        crbm->getVisibleUnitType() == UnitType::ReLU1 ||
        crbm->getVisibleUnitType() == UnitType::ReLU2 ||
        crbm->getVisibleUnitType() == UnitType::ReLU4)
    {
      v_noise.resize(size, tid);
    }

    if (crbm->getVisibleUnitType() == UnitType::Bernoulli) {
      v_rand.resize(size, tid);
    }

    vMask = zeros<value_t>(layerSize);
    vMask[seq(0,0,0,0), crbm->getMask()->size()] = *crbm->getMask();

    // pad h mask according to convolution shrinkage
    if (crbm->getConvolutionType() == ConvolutionType::Valid){
      dim_t topleft = crbm->getFilterKernelSize() / 2;
      topleft[dimCount - 1] = 0;
      hMask = zeros<value_t>(layerSize);
      hMask[topleft, originalLayerSize - 2 * topleft] = ones<value_t>(originalLayerSize - 2 * topleft);
      hMask = hMask * vMask;
    } else {
      hMask = vMask;
    }

    #pragma omp master
    {
      vbMaskSize = cb.size();
      if (getShareBiasTerms()) {
        vbMaskSize[0] = 1;
        vbMaskSize[1] = 1;
        vbMaskSize[2] = 1;
      }

      hbMaskSize = cc[0]->size();
      spMaskSize = cc[0]->size();
      if (getShareBiasTerms()) {
        hbMaskSize[0] = 1;
        hbMaskSize[1] = 1;
        hbMaskSize[2] = 1;
      }
      spMaskSize[0] = 1;
      spMaskSize[1] = 1;
      spMaskSize[2] = 1;
    }

    #pragma omp barrier

    #pragma omp master
    dlog() << "Trainer initialized. Starting training.";

    START
    for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
      #pragma omp master
      {
        error = 0;

        if (iEpoch < getMomentumDecayEpochs()) {
          const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
          momentum = (1 - t) * initialmomentum + t * finalmomentum;
        } else {
          momentum = finalmomentum;
        }
      }

      // Momentum is read by all threads therefore wait here until the master has done his work
      #pragma omp barrier

      // Make the dropout decision
#ifdef DROPOUT
      if (getDropoutMethod() != DropoutMethod::NoDrop && getDropoutStage() == DropoutStage::Epoch) {

        // Decide which units and filters to drop
        #pragma omp master
        {
          for (size_t i = 0; i < dropFilter.size(); ++i)
            dropFilter[i] = (value_t)rand() / (value_t)RAND_MAX < getFilterDropout();
          drop_master = h_rand > getHiddenDropout();
        }
        cudaStreamSynchronize(0);
        #pragma omp barrier
        if (getDropoutMethod() == DropoutMethod::DropColumn) {
          for (size_t k = tid; k < cF.size(); k += gpuCount)
            *drops[k] = drop_master;
        } else {
          for (size_t k = tid; k < cF.size(); k += gpuCount)
            *drops[k] = h_rand > getHiddenDropout();
        }
        #pragma omp barrier
      }
#endif

      for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {
#ifndef SPREAD_IMAGES
        #pragma omp master
#endif
        {
          batchError = 0;
          cbinc = momentum * cbinc;
        }

#ifdef SPREAD_IMAGES
        for (size_t k = 0; k < cF.size(); ++k) {
#else
        for (size_t k = tid; k < cF.size(); k += gpuCount) {
#endif
          *cFinc[k] = momentum * *cFinc[k] - weightcost * *cF[k];
          *ccinc[k] = momentum * *ccinc[k];
        }

        // Make the dropout decision
#ifdef DROPOUT
        if (getDropoutMethod() != DropoutMethod::NoDrop && getDropoutStage() == DropoutStage::Batch) {

          // Decide which units and filters to drop
          #pragma omp master
          {
            for (size_t i = 0; i < dropFilter.size(); ++i)
              dropFilter[i] = (value_t)rand() / (value_t)RAND_MAX < getFilterDropout();
            drop_master = h_rand > getHiddenDropout();
          }
          cudaStreamSynchronize(0);
          #pragma omp barrier

          if (getDropoutMethod() == DropoutMethod::DropColumn) {
            for (size_t k = tid; k < cF.size(); k += gpuCount)
              *drops[k] = drop_master;
          } else {
            for (size_t k = tid; k < cF.size(); k += gpuCount)
              *drops[k] = h_rand > getHiddenDropout();
          }
          #pragma omp barrier
        }
#endif

#ifdef SPREAD_IMAGES
        for (size_t iSample = tid; iSample < batchSize; iSample += gpuCount) {
#else
        for (size_t iSample = 0; iSample < batchSize; ++iSample) {
#endif
          // Make the dropout decision
#ifdef DROPOUT
          if (getDropoutMethod() != DropoutMethod::NoDrop && getDropoutStage() == DropoutStage::Sample) {

            // Decide which units and filters to drop
            #pragma omp master
            {
              for (size_t i = 0; i < dropFilter.size(); ++i)
                dropFilter[i] = (value_t)rand() / (value_t)RAND_MAX < getFilterDropout();
              drop_master = h_rand > getHiddenDropout();
            }
            cudaStreamSynchronize(0);
            #pragma omp barrier

            if (getDropoutMethod() == DropoutMethod::DropColumn) {
              for (size_t k = tid; k < cF.size(); k += gpuCount)
                *drops[k] = drop_master;
            } else {
              for (size_t k = tid; k < cF.size(); k += gpuCount)
                *drops[k] = h_rand > getHiddenDropout();
            }
            #pragma omp barrier
          }
#endif

          // Prepare CD iterations

#ifdef SPREAD_IMAGES
          if (getRandomizeTraining())
            cvneg = *cX[rand() % cX.size()];
          else
            cvneg = *cX[iSample + iBatch * batchSize];
#else
          cudaStreamSynchronize(0);
          #pragma omp barrier

          // get v
          #pragma omp master
          {
            if (getRandomizeTraining())
              *cvneg_master = *cX[rand() % cX.size()];
            else
              *cvneg_master = *cX[iSample + iBatch * batchSize];
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier
#endif

          for (size_t iCd = 0; iCd <= getCdIterations(); ++iCd) {

#ifdef SPREAD_IMAGES
            cv = cvneg;
            cvneg = zeros<complex_t>(cv.size(), cv.fullsize());
#else
            #pragma omp master
            {
              *cv_master = *cvneg_master;
              *cvneg_master = zeros<complex_t>(cv_master->size(), cv_master->fullsize());
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            if (tid != 0)
              cv = *cv_master;
            cvneg = zeros<complex_t>(cv.size(), cv.fullsize());
#endif

#ifdef SPREAD_IMAGES
            for (size_t k = 0; k < cF.size(); ++k) {
#else
            for (size_t k = tid; k < cF.size(); k += gpuCount) {
#endif
              // There appears to be a problem with the GCC implementation of OpenMP
              // that causes a syntax error when the continue statement is used.
#ifdef DROPOUT
              if (!dropFilter[k])
#endif
              {

                /*** BEGIN OF POSITIVE PHASE ***/

                // h_k = sigm(~F_k * v + c)
                ch = conj_mult_sum(cv, *cF[k]);
                if (getDbmLayer() == DbmLayer::VisibleLayer)
                  ch = 2 * ch;

                ch = ch + *cc[k];
                h2 = ifft(ch, dimCount - 1, iplan_h);

                switch (crbm->getHiddenUnitType()) {
                  case UnitType::Bernoulli: h = sigm(h2); break;
                  case UnitType::ReLU:      h = max(0.0, h2);  break;
                  case UnitType::MyReLU:    h = nrelu_mean(h2); break;
                  case UnitType::ReLU1:     h = min(1.0, max(0.0, h2));  break;
                  case UnitType::ReLU2:     h = min(2.0, max(0.0, h2));  break;
                  case UnitType::ReLU4:     h = min(4.0, max(0.0, h2));  break;
                  case UnitType::ReLU8:     h = min(8.0, max(0.0, h2));  break;
                  default:
                    dlog(Severity::Warning) << "Unsupported hidden unit type: " << crbm->getVisibleUnitType();
                }
#ifdef DROPOUT
                if (getDropoutMethod() != DropoutMethod::NoDrop)
                  h = h * *drops[k] / (1. - getHiddenDropout()) * repeat(hMask, h.size() / hMask.size());
                else
#endif
                  h = h * repeat(hMask, h.size() / hMask.size());

                // dF_k = ~h * v
                ch = fft(h, dimCount - 1, plan_h);

                if (iCd == 0) {

                  /* POSITIVE PHASE */

//                  *cFinc[k] = *cFinc[k] + epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;
                  *cFinc[k] += conj_repeat_mult(cv, ch, epsilonw);
                  *ccinc[k] = *ccinc[k] + epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
                  switch(getSparsityMethod()) {
                  case SparsityMethod::WeightsAndBias:
                    chdiff = getSparsityTarget() * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
                    *cFinc[k] = *cFinc[k] + epsilonsw * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
                    *ccinc[k] = *ccinc[k] + epsilonsb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
                    break;

                  case SparsityMethod::OnlyBias:
                    chdiff = getSparsityTarget() * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
                    *ccinc[k] = *ccinc[k] + epsilonsb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
                    break;

                  case SparsityMethod::OnlySharedBias:
                    *ccinc[k] = *ccinc[k] + epsilonsb * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (getSparsityTarget() * h.count() + -ch);
                    break;
                  }
                } else if (iCd == getCdIterations()) {

                  /* NEGATIVE PHASE */

//                  *cFinc[k] = *cFinc[k] - epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;
                  *cFinc[k] += conj_repeat_mult(cv, ch, -epsilonw);
                  *ccinc[k] = *ccinc[k] - epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
                }

                /*** RECONSTRUCT FROM SAMPLES ***/

                if (iCd < getCdIterations()) {
                  // Sample hidden states
                  switch (crbm->getHiddenUnitType()) {
                    case UnitType::Bernoulli: h = h > h_rand; break;
                    case UnitType::MyReLU:
                    case UnitType::ReLU:      h = max(0.0, h2 + sqrt(sigm(h2)) * h_noise); break;
                    case UnitType::ReLU1:     h = min(1.0, max(0.0, h2 + (h2 > 0) * (h2 < 1.0) * h_noise)); break;
                    case UnitType::ReLU2:     h = min(2.0, max(0.0, h2 + (h2 > 0) * (h2 < 2.0) * h_noise)); break;
                    case UnitType::ReLU4:     h = min(4.0, max(0.0, h2 + (h2 > 0) * (h2 < 4.0) * h_noise)); break;
                    case UnitType::ReLU8:     h = min(8.0, max(0.0, h2 + (h2 > 0) * (h2 < 8.0) * h_noise)); break;
                    default:
                      dlog(Severity::Warning) << "Unsupported hidden unit type: " << crbm->getVisibleUnitType();
                  }
#ifdef DROPOUT
                  if (getDropoutMethod() != DropoutMethod::NoDrop)
                    h = h * *drops[k] / (1. - getHiddenDropout()) * repeat(hMask, h.size() / hMask.size());
                  else
#endif
                    h = h * repeat(hMask, h.size() / hMask.size());

                  ch = fft(h, dimCount - 1, plan_h);

                  // dvneg = F * h
//                  if (getDbmLayer() == DbmLayer::TopLayer)
//                    cvneg = cvneg + 2 * *cF[k] * repeat(ch, cF[k]->size() / ch.size());
//                  else
//                    cvneg = cvneg + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
                  cvneg += repeat_mult_sum(ch, *cF[k]);
                }
              } /* drop */
            } /* end of filters */

#ifdef SPREAD_IMAGES
            if (iCd == 0)
              cbinc = cbinc + epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;
            else if (iCd == getCdIterations())
              cbinc = cbinc - epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;
#else
            #pragma omp master
            {
              //binc = binc + epsilonvb * sum(v);
              if (iCd == 0)
                cbinc = cbinc + epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;
              else if (iCd == getCdIterations())
                cbinc = cbinc - epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier
#endif

            if (iCd < getCdIterations()) {

#ifndef SPREAD_IMAGES
              // Add up local copies
              #pragma omp critical
              {
                if (tid != 0)
                  *cvneg_master = *cvneg_master + cvneg;
                cudaStreamSynchronize(0);
              }
              #pragma omp barrier
#endif

              /*** END OF POSITIVE PHASE ***/

#ifndef SPREAD_IMAGES
              #pragma omp master
#endif
              {
                if (getCalculateError() && iCd == 0)
                  v = ifft(cv, dimCount - 1, iplan_v);

                /*** BEGIN OF NEGATIVE PHASE ***/

#ifdef SPREAD_IMAGES
                if (getDbmLayer() == DbmLayer::TopLayer)
                  cvneg = 2 * cvneg + cb;
                else
                  cvneg = cvneg + cb;

                vneg = ifft(cvneg, dimCount - 1, iplan_v);
#else
                if (getDbmLayer() == DbmLayer::TopLayer)
                  *cvneg_master = 2 * *cvneg_master + cb;
                else
                  *cvneg_master = *cvneg_master + cb;

                vneg = ifft(*cvneg_master, dimCount - 1, iplan_v);
#endif

                switch (crbm->getVisibleUnitType()) {
                  case UnitType::Gaussian:  break;
                  case UnitType::Bernoulli: vneg = sigm(vneg) > v_rand; break;
                  case UnitType::MyReLU:
                  case UnitType::ReLU:      vneg = max(0.0, vneg + sqrt(sigm(vneg)) * v_noise); break;
                  case UnitType::ReLU1:     vneg = min(1.0, max(0.0, vneg + (vneg > 0) * (vneg < 1.0) * v_noise)); break;
                  case UnitType::ReLU2:     vneg = min(2.0, max(0.0, vneg + (vneg > 0) * (vneg < 2.0) * v_noise)); break;
                  case UnitType::ReLU4:     vneg = min(4.0, max(0.0, vneg + (vneg > 0) * (vneg < 4.0) * v_noise)); break;
                  case UnitType::ReLU8:     vneg = min(8.0, max(0.0, vneg + (vneg > 0) * (vneg < 8.0) * v_noise)); break;

                  default:
                    dlog(Severity::Warning) << "Unsupported visible unit type: " << crbm->getVisibleUnitType();
                }
                vneg = vneg * repeat(vMask, size / layerSize);

#ifdef SPREAD_IMAGES
                cvneg = fft(vneg, dimCount - 1, plan_v);
                if (getCalculateError() && iCd == getCdIterations() - 1) {
                  batchError += sqrt(dot((vneg - v) * repeat(hMask, size / layerSize), (vneg - v) * repeat(hMask, size / layerSize)) / voxelCount);
                }
#else
                *cvneg_master = fft(vneg, dimCount - 1, plan_v);
                if (getCalculateError() && iCd == getCdIterations() - 1) {
                  batchError += sqrt(dot((vneg - v) * repeat(vMask, size / layerSize), (vneg - v) * repeat(vMask, size / layerSize)) / voxelCount);
                }
                cudaStreamSynchronize(0);
#endif
              }

#ifndef SPREAD_IMAGES
              // Wait until master is done and copy result of cvneg_master to local thread copies
              #pragma omp barrier
#endif
            }
          } /* end of cd iterations */
        } /* end of sample */

#ifdef SPREAD_IMAGES

        cudaStreamSynchronize(0);
        #pragma omp barrier

        for (size_t k = 0; k < cF.size(); ++k) {
#else
        for (size_t k = tid; k < cF.size(); k += gpuCount) {
#endif
          // Mask filters
          for (int j = 0; j < filterBatchLength; ++j) {
            cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()];
            cv.set_fullsize(cX[0]->fullsize());
            f = ifft(cv, dimCount - 1, iplan_v);
            f = f * mask<value_t>(f.size(), crbm->getFilterKernelSize());
            cv = fft(f, dimCount - 1, plan_v);
            (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()] = cv;
          }
#ifdef SPREAD_IMAGES
          #pragma omp critical
          {
            *cF_master[k] = *cF_master[k] + *cFinc[k];
            *cc_master[k] = *cc_master[k] + *ccinc[k];
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          if (tid != 0) {
            *cF[k] = *cF_master[k];
            *cc[k] = *cc_master[k];
          }
#else
          *cF[k] = *cF[k] + *cFinc[k];
          *cc[k] = *cc[k] + *ccinc[k];
#endif
        }

#ifdef SPREAD_IMAGES
        if (getShareBiasTerms()) {
          const int channelsPerBlock = getChannelsPerBlock();
          for (int i = 0; i < size[3]; i += channelsPerBlock)
            cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)] = ones<complex_t>(1,1,1,channelsPerBlock) * sum(cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)]) * (1.f / (float)channelsPerBlock);
        }

        #pragma omp critical
        {
          *cb_master = *cb_master + cbinc;
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        if (tid != 0)
          cb = *cb_master;

        #pragma omp critical
        error += batchError;
#else
        #pragma omp master
        {
          if (getShareBiasTerms()) {
            const int channelsPerBlock = getChannelsPerBlock();
            for (int i = 0; i < size[3]; i += channelsPerBlock)
              cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)] = ones<complex_t>(1,1,1,channelsPerBlock) * sum(cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)]) * (1.f / (float)channelsPerBlock);
          }

          cb = cb + cbinc;
          error += batchError;
        }
#endif

        #pragma omp master
        {
          if (monitor)
            monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
        }

        // Just to be sure that everything is synchronized
        cudaStreamSynchronize(0);
        #pragma omp barrier
      } /* end of batch */

#ifdef MONITOR_TRAINING
      if (getUpdateModel() && (iEpoch % getUpdateModel() == 0)) {
        tensor_t hb, p, k;
        for (size_t i = tid; i < cF.size(); i += gpuCount) {
          dim_t topleft = size / 2 - crbm->getFilterKernelSize() / 2;

          f = ifft(*cF[i], dimCount - 1, iplan_v);
          p = fftshift(f, dimCount - 1);
          k = p[topleft, crbm->getFilterKernelSize()];
          if (getDbmLayer() == DbmLayer::IntermediateLayer)
            filters->at(i) = boost::make_shared<host_tensor_t>(0.5 * k);
          else
            filters->at(i) = boost::make_shared<host_tensor_t>(k);

          hb = ifft(*cc[i], dimCount - 1, iplan_h);
          hb = hb * (abs(hb) > 1e-16);
          c->at(i) = boost::make_shared<host_tensor_t>(hb);

          f = ifft(*cFinc[i], dimCount - 1, iplan_v);
          p = fftshift(f, dimCount - 1);
          k = p[topleft, crbm->getFilterKernelSize()];
          filters_inc->at(i) = boost::make_shared<host_tensor_t>(k);

          hb = ifft(*ccinc[i], dimCount - 1, iplan_h);
          hb = hb * (abs(hb) > 1e-16);
          c_inc->at(i) = boost::make_shared<host_tensor_t>(hb);
        }
        #pragma omp barrier

        #pragma omp master
        {
          f = ifft(cb, dimCount - 1, iplan_v);
          f = f * (abs(f) > 1e-16);
          b = boost::make_shared<host_tensor_t>(f);
          crbm->setVisibleBias(b);
          newState->setModel(crbm);
          newState->setCurrentEpoch(iEpoch);

          f = ifft(cbinc, dimCount - 1, iplan_v);
          f = f * (abs(f) > 1e-16);
          b_inc = boost::make_shared<host_tensor_t>(f);
          inc->setVisibleBias(b_inc);
          newState->setModelIncrement(inc);
        }
      }
      #pragma omp barrier
#endif

      #pragma omp master
      {
        if (getCalculateError())
          dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / cX.size();
        else
          dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

        if (monitor)
          monitor->reportProgress(100. * (iEpoch + 1) / epochCount, getUpdateModel() && (iEpoch % getUpdateModel() == 0));

        epsilonw *= getLearningDecay();
        epsilonsw *= getLearningDecay();
        epsilonvb *= getLearningDecay();
        epsilonhb *= getLearningDecay();
        epsilonsb *= getLearningDecay();
      }
    } /* end of epochs */

    #pragma omp master
    {
      newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
      newState->setReconstructionError(error / cX.size());
      cX.clear();
    }

    // Free up memory
#ifdef SPREAD_IMAGES
    for (size_t k = 0; k < cF.size(); ++k) {
#else
    for (size_t k = tid; k < cF.size(); k += gpuCount) {
#endif
      cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
#ifdef DROPOUT
      drops[k] = boost::shared_ptr<tensor_t>();
#endif
    }

    {
      tensor_t hb, p, k;
#ifdef SPREAD_IMAGES
      for (size_t i = 0; i < cF.size(); ++i) {
        #pragma omp master
#else
      for (size_t i = tid; i < cF.size(); i += gpuCount) {
#endif
        {
          for (int j = 0; j < filterBatchLength; ++j) {
            dim_t topleft = size / 2 - crbm->getFilterKernelSize() / 2, fullsize = cv.fullsize();
            cv = (*cF[i])[seq(0,0,0,j*cv.size()[3]), cv.size()];
            cv.set_fullsize(fullsize);
            f = ifft(cv, dimCount - 1, iplan_v);
            p = fftshift(f, dimCount - 1);
            k = p[topleft, crbm->getFilterKernelSize()];

            if (getDbmLayer() == DbmLayer::IntermediateLayer)
              filters->at(i * filterBatchLength + j) = boost::make_shared<host_tensor_t>(0.5 * k);
            else
              filters->at(i * filterBatchLength + j) = boost::make_shared<host_tensor_t>(k);
          }


          h = ifft(*cc[i], dimCount - 1, iplan_h);
          h = h * (abs(h) > 1e-16);

          for (int j = 0; j < filterBatchLength; ++j) {
            hb = h[seq(0,0,0,j),originalLayerSize];
            c->at(i * filterBatchLength + j) = boost::make_shared<host_tensor_t>(hb);
          }
        }

        #pragma omp master
        cF_master[i] = cc_master[i] = boost::shared_ptr<ctensor_t>();

        cF[i] = cc[i] = boost::shared_ptr<ctensor_t>();
      }

      #pragma omp master
      {
        f = ifft(cb, dimCount - 1, iplan_v);
        f = f * (abs(f) > 1e-16);
        b = boost::make_shared<host_tensor_t>(f[seq(0,0,0,0), originalSize]);
        crbm->setVisibleBias(b);
      }
    }

    cudaDeviceSynchronize();
    #pragma omp barrier

    if (tid < deviceCount) {
      if (tid == 0) {
        for (int i = 1; i < min(deviceCount, gpuCount); ++i)
          cudaDeviceDisablePeerAccess(i);
      } else {
        cudaDeviceDisablePeerAccess(0);
      }
    }
  } /* end of parallel code */
  newState->setModel(crbm);
#endif
}

}

}
