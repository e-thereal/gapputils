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
#include <tbblas/ones.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/io.hpp>
#include <tbblas/mask.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/shift.hpp>

#include <boost/timer.hpp>

#include <fstream>
#include <omp.h>

#include "math.hpp"

#include "TensorWriter.h"

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
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);

  CHECK_MEMORY_LAYOUT2(SparsityMethod, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);

  CHECK_MEMORY_LAYOUT2(LearningRate, trainer);
  CHECK_MEMORY_LAYOUT2(LearningDecay, trainer);
  CHECK_MEMORY_LAYOUT2(InitialMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(FinalMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(MomentumDecayEpochs, trainer);
  CHECK_MEMORY_LAYOUT2(WeightDecay, trainer);
  CHECK_MEMORY_LAYOUT2(WeightVectorLimit, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
  CHECK_MEMORY_LAYOUT2(VisibleDropout, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenDropout, trainer);
  CHECK_MEMORY_LAYOUT2(FilterDropout, trainer);
  CHECK_MEMORY_LAYOUT2(DropoutMethod, trainer);
  CHECK_MEMORY_LAYOUT2(DropoutStage, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(UpdateModel, trainer);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, trainer);
  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(ModelIncrement, trainer);
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

#define MONITOR_TRAINING

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace thrust::placeholders;

  typedef float value_t;

  const unsigned dimCount = Model::dimCount;
  typedef complex<value_t> complex_t;
  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor<complex_t, dimCount, false> host_ctensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

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
  const size_t batchCount = getTensors()->size() / batchSize;
  const size_t epochCount = getEpochCount();

  // Prepare sizes
  dim_t size = getTensors()->at(0)->size();
  dim_t layerSize = size;
  layerSize[dimCount - 1] = 1;

  size_t layerVoxelCount = 1;
  size_t voxelCount = sum(*getInitialModel()->getMask()) * size[dimCount - 1];
  for (size_t i = 0; i < dimCount - 1; ++i)
    layerVoxelCount *= layerSize[i];

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
    crbm->setMask(boost::make_shared<host_tensor_t>(*oldCrbm->getMask()));

    // Prepare an early memory clean up of the old model
    // The trainer holds to pointers to the model, hence if the use_count is 2
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
  std::vector<boost::shared_ptr<host_ctensor_t> > cX;
  {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > tensors = getTensors();
    tensor_t x, hMask;
    ctensor_t cx;
    plan_t plan_v;

    hMask = *crbm->getMask();

    for (size_t i = 0; i < tensors->size(); ++i) {
      x = *tensors->at(i);

      for (unsigned j = 0; j < dimCount - 1; ++j) {
        if (x.size()[j] != upper_power_of_two(x.size()[j])) {
          dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Aborting!";
          return;
        }
      }

      if (crbm->getVisibleUnitType() == UnitType::Gaussian)
        x = (x - crbm->getMean()) / crbm->getStddev() * repeat(hMask, size / layerSize);
      cx = fft(x, dimCount - 1, plan_v);
      cX.push_back(boost::shared_ptr<host_ctensor_t>(new host_ctensor_t(cx)));

      if (getAtomicWorkflow() && tensors.use_count() == 2)
        tensors->at(i) = boost::shared_ptr<host_tensor_t>();
    }

    // Prepare an early memory clean up of the training set
    // The trainer holds to pointers to the training set, hence if the use_count is 2
    // no other modules have any business with the training set any more
    if (getAtomicWorkflow() && tensors.use_count() == 2)
      tensors->clear();
  }
  dlog() << "FFTs precalculated.";

  std::vector<boost::shared_ptr<ctensor_t> > cF(filters->size()), cFinc(filters->size());
  std::vector<boost::shared_ptr<tensor_t> > drops(filters->size());
  std::vector<bool> dropFilter(filters->size());

  // Copy visible bias to the device
  ctensor_t cb, cbinc;
  cb.set_name("cb");
  cbinc.set_name("cbinc");
  {
    tensor_t f = *b;
    plan_t plan_v;
    if (getShareBiasTerms())
      f = ones<value_t>(f.size()) * sum(f) / f.count();
    cb = fft(f, dimCount - 1, plan_v);
    cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
  }
  b.reset();

  std::vector<boost::shared_ptr<ctensor_t> > cc(c->size()), ccinc(c->size());

  // Declare variables used for training
  tensor_t v, vneg;           // to monitor training (calculate the error)
  ctensor_t* cv_master;      // Read from the master thread and then each other thread reads from the master device
  ctensor_t* cvneg_master;   // All threads add their version to the master threads version (needs to be synchronized)
  tensor_t drop_master;
  v.set_name("v");
  vneg.set_name("vneg");

  dim_t vbMaskSize = cb.size(), hbMaskSize, spMaskSize;
  if (getShareBiasTerms()) {
    vbMaskSize[0] = 1;
    vbMaskSize[1] = 1;
    vbMaskSize[2] = 1;
  }

  value_t error = 0, batchError = 0;

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

    // FFT plans
    plan_t plan_h, iplan_h, plan_v, iplan_v;

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, k, p;   // filter, kernel, padded kernel
      ctensor_t cf;
      for (size_t i = tid; i < filters->size(); i += gpuCount) {
        k = *filters->at(i);
        dim_t topleft = size / 2 - k.size() / 2;
        p = zeros<value_t>(size);
        p[topleft, k.size()] = k;
        f = ifftshift(p, dimCount - 1);

//        f = *filters->at(i);
        cf = fft(f, dimCount - 1, plan_v);
        cF[i] = boost::make_shared<ctensor_t>(cf);
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cf.size(), cf.fullsize()));
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
      tensor_t h;
      ctensor_t ch;
      for (size_t i = tid; i < c->size(); i += gpuCount) {
        h = *c->at(i);
        if (getShareBiasTerms())
          h = ones<value_t>(h.size()) * sum(h) / h.count();
        ch = fft(h, dimCount - 1, plan_h);
        cc[i] = boost::make_shared<ctensor_t>(ch);
        ch = zeros<complex_t>(ch.size(), ch.fullsize());
        ccinc[i] = boost::make_shared<ctensor_t>(ch);
        drops[i] = boost::make_shared<tensor_t>();
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
    tensor_t h, h2, f, hMask;        // for sigm, sampling, filter masking and visible masking
    ctensor_t ch, chdiff, ch_full;
    ctensor_t cv;                   // Read from the master thread and then each other thread reads from the master device
    ctensor_t cvneg;                // All threads add their version to the master threads version (needs to be synchronized)

    #pragma omp master
    {
      cv_master = &cv;
      cvneg_master = &cvneg;
    }

    random_tensor<value_t, dimCount, true, uniform<value_t> > h_rand(layerSize, tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise(layerSize, tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(size, tid);

    hMask = *crbm->getMask();

    #pragma omp master
    {
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

      // Momentum is read by all threads therefore wait here until the master has done its work
      #pragma omp barrier

      // Make the dropout decision
      if (getDropoutStage() == DropoutStage::Epoch) {

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
            *drops[k] = drop_master; //h_rand > getHiddenDropout();
        } else {
          for (size_t k = tid; k < cF.size(); k += gpuCount)
            *drops[k] = h_rand > getHiddenDropout();
        }
        #pragma omp barrier
      }

      for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {
        #pragma omp master
        batchError = 0;

        for (size_t k = tid; k < cF.size(); k += gpuCount) {
          *cFinc[k] = momentum * *cFinc[k] - weightcost * *cF[k];
          *ccinc[k] = momentum * *ccinc[k];
        }

        #pragma omp master
        cbinc = momentum * cbinc;

        // Make the dropout decision
        if (getDropoutStage() == DropoutStage::Batch) {

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

        for (size_t iSample = 0; iSample < batchSize; ++iSample) {

          cudaStreamSynchronize(0);
          #pragma omp barrier

          // get v
          #pragma omp master
          {
            if (getRandomizeTraining())
              *cv_master = *cX[rand() % cX.size()];
            else
              *cv_master = *cX[iSample + iBatch * batchSize];
            *cvneg_master = zeros<complex_t>(cv_master->size(), cv_master->fullsize());
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          if (tid != 0)
            cv = *cv_master;
          cvneg = zeros<complex_t>(cv.size(), cv.fullsize());

          // Make the dropout decision
          if (getDropoutStage() == DropoutStage::Sample) {

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

          for (size_t k = tid; k < cF.size(); k += gpuCount) {

            // There appears to be a problem with the GCC implementation of OpenMP
            // that causes a syntax error when the continue statement is used.

            if (!dropFilter[k]) {

              /*** BEGIN OF POSITIVE PHASE ***/

              // h_k = sigm(~F_k * v + c)
              ch_full = conj(*cF[k]) * cv;
              ch = sum(ch_full, dimCount - 1);
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
              h = h * *drops[k] / (1. - getHiddenDropout()) * hMask;

              // dF_k = ~h * v
              ch = fft(h, dimCount - 1, plan_h);
              *cFinc[k] = *cFinc[k] + epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;
              *ccinc[k] = *ccinc[k] + epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;

              switch(getSparsityMethod()) {
              case SparsityMethod::WeightsAndBias:
                chdiff = getSparsityTarget() * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
                *cFinc[k] = *cFinc[k] + epsilonsw * repeat(conj(chdiff), cv.size() / ch.size()) * cv;
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
              h = h * *drops[k] / (1. - getHiddenDropout()) * hMask;

              ch = fft(h, dimCount - 1, plan_h);

              /*** BEGIN OF NEGATIVE PHASE ***/

              // dvneg = F * h
              cvneg = cvneg + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
            }
          }

          // Add up local copies
          #pragma omp critical
          {
            if (tid != 0)
              *cvneg_master = *cvneg_master + cvneg;
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          /*** END OF POSITIVE PHASE ***/

          #pragma omp master
          {
            if (getCalculateError())
              v = ifft(cv, dimCount - 1, iplan_v);

            //binc = binc + epsilonvb * sum(v);
            cbinc = cbinc + epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

            /*** END OF NEGATIVE PHASE ***/

            *cvneg_master = *cvneg_master + cb;

            vneg = ifft(*cvneg_master, dimCount - 1, iplan_v);

            switch (crbm->getVisibleUnitType()) {
              case UnitType::Gaussian: break;
//              case UnitType::Bernoulli: vneg = sigm(vneg); break;
              case UnitType::MyReLU:
              case UnitType::ReLU:      vneg = max(0.0, vneg + sqrt(sigm(vneg)) * v_noise); break;
              case UnitType::ReLU1:     vneg = min(1.0, max(0.0, vneg + (vneg > 0) * (vneg < 1.0) * v_noise)); break;
              case UnitType::ReLU2:     vneg = min(2.0, max(0.0, vneg + (vneg > 0) * (vneg < 2.0) * v_noise)); break;
              case UnitType::ReLU4:     vneg = min(4.0, max(0.0, vneg + (vneg > 0) * (vneg < 4.0) * v_noise)); break;
              case UnitType::ReLU8:     vneg = min(8.0, max(0.0, vneg + (vneg > 0) * (vneg < 8.0) * v_noise)); break;

              default:
                dlog(Severity::Warning) << "Unsupported visible unit type: " << crbm->getVisibleUnitType();
            }
            vneg = vneg * repeat(hMask, size / layerSize);
            *cvneg_master = fft(vneg, dimCount - 1, plan_v);

            if (getCalculateError()) {
              batchError += sqrt(dot((vneg - v) * repeat(hMask, size / layerSize), (vneg - v) * repeat(hMask, size / layerSize)) / voxelCount);
            }
            cudaStreamSynchronize(0);
          }

          // Wait until master is done and copy result of cvneg_master to local thread copies
          #pragma omp barrier

          if (tid != 0)
            cvneg = *cvneg_master;

          cudaStreamSynchronize(0);
          #pragma omp barrier

          for (size_t k = tid; k < cF.size(); k += gpuCount) {
            if (!dropFilter[k]) {

              // h_k = sigm(~F_k * v + c)
              ch_full = conj(*cF[k]) * cvneg;
              ch = sum(ch_full, dimCount - 1);
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
                  dlog(Severity::Warning) << "Unsupported hidden unit type: " << crbm->getHiddenUnitType();
              }
              h = h * *drops[k] / (1. - getHiddenDropout()) * hMask;

              // dF_k = ~h * v
              ch = fft(h, dimCount - 1, plan_h);
              *cFinc[k] = *cFinc[k] - epsilonw * repeat(conj(ch), cvneg.size() / ch.size()) * cvneg;
              *ccinc[k] = *ccinc[k] - epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
            }
          }

          //binc = binc - epsilonvb * sum(vneg);
          #pragma omp master
          cbinc = cbinc -  epsilonvb * mask<complex_t>(cvneg.size(), cvneg.fullsize(), vbMaskSize) * cvneg;
        } /* end of sample */

        for (size_t k = tid; k < cF.size(); k += gpuCount) {
          f = ifft(*cFinc[k], dimCount - 1, iplan_v);
          f = f * mask<value_t>(f.size(), crbm->getFilterKernelSize());
          *cFinc[k] = fft(f, dimCount - 1, plan_v);
          *cF[k] = *cF[k] + *cFinc[k];
          *cc[k] = *cc[k] + *ccinc[k];
        }
        #pragma omp master
        {

          if (getShareBiasTerms()) {
            const int channelsPerBlock = getChannelsPerBlock();
            for (int i = 0; i < size[3]; i += channelsPerBlock)
              cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)] = ones<complex_t>(1,1,1,channelsPerBlock) * sum(cbinc[seq(0,0,0,i), seq(1,1,1,channelsPerBlock)]) * (1.f / (float)channelsPerBlock);
          }

          cb = cb + cbinc;

          if (monitor)
            monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));

          error += batchError;
        }


      } /* end of batch */

#ifdef MONITOR_TRAINING
      if (getUpdateModel() && (iEpoch % getUpdateModel() == 0)) {
        tensor_t hb, p, k;
        for (size_t i = tid; i < cF.size(); i += gpuCount) {
          dim_t topleft = size / 2 - crbm->getFilterKernelSize() / 2;

          f = ifft(*cF[i], dimCount - 1, iplan_v);
          p = fftshift(f, dimCount - 1);
          k = p[topleft, crbm->getFilterKernelSize()];
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
    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
      drops[k] = boost::shared_ptr<tensor_t>();
    }

    {
      tensor_t hb, p, k;
      for (size_t i = tid; i < cF.size(); i += gpuCount) {
        dim_t topleft = size / 2 - crbm->getFilterKernelSize() / 2;

        f = ifft(*cF[i], dimCount - 1, iplan_v);
        p = fftshift(f, dimCount - 1);
        k = p[topleft, crbm->getFilterKernelSize()];
        filters->at(i) = boost::make_shared<host_tensor_t>(k);

        hb = ifft(*cc[i], dimCount - 1, iplan_h);
        hb = hb * (abs(hb) > 1e-16);
        c->at(i) = boost::make_shared<host_tensor_t>(hb);

        cF[i] = cc[i] = boost::shared_ptr<ctensor_t>();
      }

      #pragma omp master
      {
        f = ifft(cb, dimCount - 1, iplan_v);
        f = f * (abs(f) > 1e-16);
        b = boost::make_shared<host_tensor_t>(f);
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
}

}

}
