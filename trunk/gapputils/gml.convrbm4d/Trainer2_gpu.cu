/*
 * Trainer2_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer2.h"

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
#include <tbblas/filter.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/filter2.hpp>

#include <boost/timer.hpp>

#include <fstream>
#include <omp.h>

#include <convnet/nvmatrix.cuh>
#include <convnet/cudaconv2.cuh>

#include "math.hpp"

// All monitoring code is enclosed by #ifdef MONITOR_TRAINING blocks
//#define MONITOR_TRAINING

namespace gml {

namespace convrbm4d {

Trainer2Checker::Trainer2Checker() {
  Trainer2 trainer;
  trainer.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);
  CHECK_MEMORY_LAYOUT2(FilterMethod, trainer);
  CHECK_MEMORY_LAYOUT2(Stride, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateW, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateVB, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateHB, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityMethod, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
//  CHECK_MEMORY_LAYOUT2(Logfile, trainer);

  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(AverageEpochTime, trainer);
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

unsigned int upper_power_of_two(unsigned int v);

void Trainer2::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace thrust::placeholders;

  typedef float value_t;

  const unsigned dimCount = Model::dimCount;
//  typedef complex<value_t> complex_t;
//  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
//  typedef tensor<complex_t, dimCount, true> ctensor_t;
//  typedef tensor<complex_t, dimCount, false> host_ctensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  const int gpuCount = getGpuCount();

  if (getShareBiasTerms()) {
    dlog(Severity::Warning) << "Shared hidden bias terms are not supported in this release. Hidden bias terms are treated as not shared.";
  }

  if (getSparsityMethod() != SparsityMethod::OnlySharedBias) {
    dlog(Severity::Warning) << "Only '" << SparsityMethod::OnlySharedBias << "' is supported as a sparsity method. Sparsity method will be treated as 'OnlySharedBias'.";
  }

  if (getFilterMethod() == FilterMethod::ConvNet && (getTensors()->at(0)->size()[0] / getStride() > 128 || getTensors()->at(0)->size()[1] / getStride() > 128 || getTensors()->at(0)->size()[2] > 1)) {
    dlog(Severity::Warning) << "Invalid input dimension. Only 2D images with a maximum resolution of 128 x 128 are supported. Aborting!";
//    return;
  }

//  if (deviceCount < gpuCount) {
//    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << gpuCount << " are required according to GpuCount. Aborting!";
//    return;
//  }

  assert(omp_get_num_threads() == 1);

  cudaSetDevice(0);
  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

#ifdef MONITOR_TRAINING
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugVisibleBiases(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugHiddenBiases(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugHiddenUnits(
      new std::vector<boost::shared_ptr<host_tensor_t> >());
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugReconstructions(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  std::ofstream logfile;
  value_t bmean, bsd, cmean, csd, Fmean, Fsd, hmean, vnegmean;
  ctensor_t dcv_master;
#endif

  /*** PREPARE MASTER THREAD ***/

  boost::shared_ptr<Model> crbm = getInitialModel()->clone();

  std::vector<boost::shared_ptr<host_tensor_t> >& tensors = *getTensors();
  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  const size_t epochCount = getEpochCount();
  const int stride = getStride();

  std::vector<boost::shared_ptr<host_tensor_t> > X;

  // Normalize input and pre-calculate the FFT
  {
    tensor_t x;
    for (size_t i = 0; i < tensors.size(); ++i) {
      x = *tensors[i];

      for (unsigned j = 0; j < dimCount - 1; ++j) {
        if (x.size()[j] != upper_power_of_two(x.size()[j])) {
          dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Aborting!";
          return;
        }
      }

      if (crbm->getVisibleUnitType() == UnitType::Gaussian)
        x = (x - crbm->getMean()) / crbm->getStddev();
      X.push_back(boost::shared_ptr<host_tensor_t>(new host_tensor_t(x)));
    }
  }

  // Create arrays for filters
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<boost::shared_ptr<tensor_t> > F(filters.size()), Finc(filters.size());

  // Copy visible bias to the device
  host_tensor_t& vb = *crbm->getVisibleBias();
  tensor_t b, binc;
  b.set_name("b");
  binc.set_name("binc");
  {
    tensor_t f = vb;
    if (getShareBiasTerms())
      f = ones<value_t>(f.size()) * sum(f) / f.count();
    b = f;
    binc = zeros<value_t>(b.size(), b.fullsize());
  }

  std::vector<boost::shared_ptr<host_tensor_t> >& hb = *crbm->getHiddenBiases();
  std::vector<boost::shared_ptr<tensor_t> > c(hb.size()), cinc(hb.size());

  // Prepare sizes
  dim_t size = tensors[0]->size();
  dim_t layerSize = size;
  layerSize[dimCount - 1] = 1;

  size_t layerVoxelCount = 1;
  size_t voxelCount = tensors[0]->count();
  for (size_t i = 0; i < dimCount - 1; ++i)
    layerVoxelCount *= layerSize[i];

  // Initialize constants
  value_t epsilonw =  getLearningRateW() / batchSize / layerVoxelCount; // Learning rate for weights
  value_t epsilonsw =  getLearningRateW() * getSparsityWeight() / batchSize / layerVoxelCount; // Sparsity weight
  value_t epsilonvb = getLearningRateVB() / batchSize;                  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRateHB() / batchSize;                  // Learning rate for biases of hidden units
  value_t epsilonsb = getLearningRateHB() * getSparsityWeight() / batchSize;                  // Sparsity weight
  value_t weightcost = 0.0002 * getLearningRateW();
  value_t initialmomentum = 0.5; //65; // 0.5f;
  value_t finalmomentum = 0.9; // 65; // 0.9f;
  value_t momentum;

  // Declare variables used for training
//  tensor_t v, vneg;         // to monitor training (calculate the error)
  tensor_t v_master;      // Read from the master thread and then each other thread reads from the master device
  tensor_t vneg_master;   // All threads add their version to the master threads version (needs to be synchronized)
//  v.set_name("v");
//  vneg.set_name("vneg");
  v_master.set_name("v_master");
  vneg_master.set_name("vneg_master");

//  dim_t vbMaskSize = b.size(), hbMaskSize, spMaskSize;
//  if (getShareBiasTerms()) {
//    vbMaskSize[0] = 1;
//    vbMaskSize[1] = 1;
//    vbMaskSize[2] = 1;
//  }

  bool monitorTraining = false;
  value_t error = 0, batchError = 0;

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

#ifdef MONITOR_TRAINING
    tensor_t l;
    ctensor_t dcv;
#endif

    #pragma omp master
    assert(tid == 0);   // Check the assumption that the first thread is the master thread

    // FFT plans
//    plan_t plan_h, iplan_h, plan_v, iplan_v;

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, f2;
      for (size_t i = tid; i < filters.size(); i += gpuCount) {
//        f = *filters[i];
//        f2 = fftshift(f, dimCount - 1);
//        dim_t topleft = (f2.size() - crbm->getFilterKernelSize() + 1) / 2;
//        F[i] = boost::make_shared<tensor_t>(f2[topleft, crbm->getFilterKernelSize()]);
        F[i] = boost::make_shared<tensor_t>(*filters[i]);
        Finc[i] = boost::make_shared<tensor_t>(zeros<value_t>(crbm->getFilterKernelSize()));
      }
    }
    {
      tensor_t h;
      for (size_t i = tid; i < c.size(); i += gpuCount) {
        h = *hb[i];
        if (getShareBiasTerms())
          h = ones<value_t>(h.size()) * sum(h) / h.count();
        c[i] = boost::make_shared<tensor_t>(h);
        h = zeros<value_t>(h.size(), h.fullsize());
        cinc[i] = boost::make_shared<tensor_t>(h);
      }
    }

    // Declare variables used for training
    tensor_t h, h2, f;       // for sigm, sampling and masking
    tensor_t hdiff, h_full;
    tensor_t v, v2;         // Read from the master thread and then each other thread reads from the master device
    tensor_t vneg;      // All threads add their version to the master threads version (needs to be synchronized)
    random_tensor<value_t, dimCount, true, uniform<value_t> > h_rand(layerSize, tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise(layerSize, tid);

    #pragma omp master
    {
//      hbMaskSize = c[0]->size();
//      spMaskSize = c[0]->size();
//      if (getShareBiasTerms()) {
//        hbMaskSize[0] = 1;
//        hbMaskSize[1] = 1;
//        hbMaskSize[2] = 1;
//      }
//      spMaskSize[0] = 1;
//      spMaskSize[1] = 1;
//      spMaskSize[2] = 1;
    }

    #pragma omp barrier

    #pragma omp master
    dlog() << "Trainer2 initialized. Starting training.";

  #ifdef MONITOR_TRAINING
    #pragma omp master
    monitorTraining = getLogfile().size();
    #pragma omp barrier

    if (monitorTraining) {
      #pragma omp master
      {
        logfile.open(getLogfile().c_str());

        logfile << "b.mean, b.sd, c.mean, c.sd, F.mean, F.sd, h.mean, vneg.mean, error" << std::endl;

        bmean = bsd = cmean = csd = Fmean = Fsd = hmean = vnegmean = 0;

        f = ifft(cb, dimCount - 1, iplan_v);
        bmean = sum(f) / f.count();
      }
      value_t tc = 0, tF = 0;
      for (size_t k = tid; k < filters.size(); k += gpuCount) {
        l = ifft(*cc[k], dimCount - 1, iplan_h);
        tc += sum(l) / l.count();
        f = ifft(*cF[k], dimCount - 1, iplan_v);
        tF += sum(f) / f.count();
      }
      #pragma omp critical
      {
        cmean += tc;
        Fmean += tF;
      }
      #pragma omp barrier
      #pragma omp master
      {
        cmean /= filters.size();
        Fmean /= filters.size();

        f = ifft(cb, dimCount - 1, iplan_v);
        bsd = sqrt(dot(f - bmean, f - bmean) / f.count());
      }
      tc = tF = 0;
      for (size_t k = tid; k < filters.size(); k += gpuCount) {
        l = ifft(*cc[k], dimCount - 1, iplan_h);
        tc += dot(l - cmean, l - cmean) / l.count();
        f = ifft(*cF[k], dimCount - 1, iplan_v);
        tF += dot(f - Fmean, f - Fmean) / f.count();
      }
      #pragma omp critical
      {
        csd += tc;
        Fsd += tF;
      }
      #pragma omp barrier
      #pragma omp master
      {
        csd = sqrt(csd / filters.size());
        Fsd = sqrt(Fsd / filters.size());

        logfile << bmean << ", " << bsd << ", " << cmean << ", " << csd << ", " << Fmean << ", " << Fsd << ", " << hmean << ", " << vnegmean << ", " << 0 << std::endl;
      }
    }
  #endif

    NVMatrix nvVisibles, nvWeights, nvHiddens;
    if (getFilterMethod() == FilterMethod::ConvNet) {
      nvVisibles.resize(b.count(), batchSize);
      nvWeights.resize(F[0]->count(), F.size());
    }

    START
    for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
      #pragma omp master
      {
  #ifdef MONITOR_TRAINING
      debugHiddenUnits->clear();
      debugReconstructions->clear();
  #endif

        error = 0;
        momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);
      }

      // Momentum is read by all threads therefore wait here until the master has done its work
      #pragma omp barrier

      for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {
        #pragma omp master

        batchError = 0;

  #ifdef MONITOR_TRAINING
        #pragma omp master
        bmean = bsd = cmean = csd = Fmean = Fsd = hmean = vnegmean = 0;
  #endif

        for (size_t k = tid; k < F.size(); k += gpuCount) {
          *Finc[k] = momentum * *Finc[k] - weightcost * *F[k];
          *cinc[k] = momentum * *cinc[k];
        }

        #pragma omp master
        binc = momentum * binc;

        if (getFilterMethod() == FilterMethod::ConvNet) {
          /**
           * void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                         int imgSizeY, int numModulesY, int numModulesX,
                         int paddingStart, int moduleStride, int numImgColors, int numGroups);

             void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                         int imgSizeY, int numModulesY, int numModulesX, int filterSize,
                         int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum);

             void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                         int imgSizeY, int imgSizeX, int numModulesY,
                         int paddingStart, int moduleStride, int numImgColors, int numGroups);
           */

          // Positive Phase

          convFilterActs(nvVisibles, nvWeights, nvHiddens,
              b.size()[1] * stride, c[0]->size()[1], c[0]->size()[0],
              -(F[0]->size()[1] * stride - 1) / 2, stride, b.size()[3] / (stride * stride), 1);

          convWeightActs(nvVisibles, nvHiddens, nvWeights,
              b.size()[1] * stride, c[0]->size()[1], c[0]->size()[0], F[0]->size()[0] * stride,
              -(F[0]->size()[1] * stride - 1) / 2, stride, b.size()[3] / (stride * stride), 1, false);

          // Negative Phase

          convImgActs(nvHiddens, nvWeights, nvVisibles,
              b.size()[1] * stride, b.size()[0] * stride, c[0]->size()[1],
              -(F[0]->size()[1] * stride - 1) / 2, stride, b.size()[3] / (stride * stride), 1);

          convFilterActs(nvVisibles, nvWeights, nvHiddens,
              b.size()[1] * stride, c[0]->size()[1], c[0]->size()[0],
              -(F[0]->size()[1] * stride - 1) / 2, stride, b.size()[3] / (stride * stride), 1);

          convWeightActs(nvVisibles, nvHiddens, nvWeights,
              b.size()[1] * stride, c[0]->size()[1], c[0]->size()[0], F[0]->size()[0] * stride,
              -(F[0]->size()[1] * stride - 1) / 2, stride, b.size()[3] / (stride * stride), 1, false);
        }

        for (size_t iSample = 0; iSample < batchSize; ++iSample) {

          cudaStreamSynchronize(0);
          #pragma omp barrier

          // get v
          #pragma omp master
          {
            if (getRandomizeTraining())
              v_master = *X[rand() % X.size()];
            else
              v_master = *X[iSample + iBatch * batchSize];
            vneg_master = zeros<value_t>(v_master.size(), v_master.fullsize());
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          v = v_master;
          vneg = zeros<value_t>(v.size(), v.fullsize());

  #ifdef MONITOR_TRAINING
          if (iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
            #pragma omp master
            {
              dcv_master = zeros<complex_t>(cv_master.size(), cv_master.fullsize());
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier
            dcv = zeros<complex_t>(cv.size(), cv.fullsize());
          }
  #endif

          for (size_t k = tid; k < F.size(); k += gpuCount) {

            /*** BEGIN OF POSITIVE PHASE ***/

            // h_k = sigm(~F_k * v + c)
            //ch_full = conj(*cF[k]) * cv;
            switch (getFilterMethod()) {
            case FilterMethod::FFT:
              h_full = filter(v, flip(*F[k]), dimCount - 1);
              break;
            case FilterMethod::NaiveConvolution:
              h_full = filter3d(v, *F[k], naive());
              break;
            case FilterMethod::OptimizedConvolution:
              h_full = filter3d(v, *F[k], optimized());
              break;

            case FilterMethod::ConvNet:
            case FilterMethod::NoConv:
              h_full.resize(v.size(), v.fullsize());
              break;
            }

            if (getFilterMethod() != FilterMethod::ConvNet && getFilterMethod() != FilterMethod::NoConv)
              h = sum(h_full, dimCount - 1);
            else
              h.resize(layerSize, layerSize);
            h2 = h + *c[k];

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

            // dF_k = ~h * v
            if (iEpoch == 0 && iBatch == 0 && iSample == 0) {
              f = filter(v, repeat(flip(h), v.size() / h.size()), F[k]->size(), dimCount - 1);
            } else {
              switch (getFilterMethod()) {
              case FilterMethod::FFT:
                f = filter(v, repeat(flip(h), v.size() / h.size()), F[k]->size(), dimCount - 1);
                break;
              case FilterMethod::NaiveConvolution:
                h_full = filter3d(v, *F[k], naive());
                break;
              case FilterMethod::OptimizedConvolution:
                h_full = filter3d(v, *F[k], optimized());
                break;
              case FilterMethod::ConvNet:
              case FilterMethod::NoConv:
                break;
              }
            }
            *Finc[k] = *Finc[k] + epsilonw * f;
            *cinc[k] = *cinc[k] + epsilonhb * h;
            *cinc[k] = *cinc[k] + epsilonsb * (getSparsityTarget() - sum(h));

  #ifdef MONITOR_TRAINING
            if (iSample == 0 && iBatch == 0 && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
              #pragma omp critical
              debugHiddenUnits->push_back(boost::make_shared<host_tensor_t>(h));
            }

            if (iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
              dcv = dcv + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
            }

            if (monitorTraining && iSample == 0) {
              #pragma omp critical
              hmean += sum(h) / h.count() / cF.size();
            }
  #endif

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

  #ifdef MONITOR_TRAINING
            if (iSample == 0 && iBatch == 0 && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
              #pragma omp critical
              debugHiddenUnits->push_back(boost::make_shared<host_tensor_t>(h));
            }
  #endif

            /*** BEGIN OF NEGATIVE PHASE ***/

            // dvneg = F * h
            switch (getFilterMethod()) {
            case FilterMethod::FFT:
              v2 = filter(repeat(h, v.size() / h.size()), *F[k], dimCount - 1);
              break;
            case FilterMethod::NaiveConvolution:
              h_full = filter3d(v, *F[k], naive());
              v2.resize(v.size(), v.fullsize());
              break;
            case FilterMethod::OptimizedConvolution:
              h_full = filter3d(v, *F[k], optimized());
              v2.resize(v.size(), v.fullsize());
              break;
            case FilterMethod::ConvNet:
            case FilterMethod::NoConv:
              v2.resize(v.size(), v.fullsize());
              break;
            }
            vneg = vneg + v2;
          }
#if 1

          // Add up local copies
          #pragma omp critical
          {
            vneg_master = vneg_master + vneg;
#ifdef MONITOR_TRAINING
            if (iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0)
              dcv_master = dcv_master + dcv;
#endif
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          /*** END OF POSITIVE PHASE ***/

          #pragma omp master
          {

            binc = binc + epsilonvb * v;

            /*** END OF NEGATIVE PHASE ***/

            vneg_master = vneg_master + b;

            switch(crbm->getVisibleUnitType()) {
              case UnitType::Bernoulli:
                vneg_master = sigm(vneg_master);
                break;

              case UnitType::Gaussian:
              break;

              default:
                dlog(Severity::Warning) << "Unsupported visible unit type: " << crbm->getVisibleUnitType();
            }

    #ifdef MONITOR_TRAINING
            if (monitorTraining && iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
              dcv_master = dcv_master + cb;
              f = ifft(dcv_master, dimCount - 1, iplan_v);
              if (crbm->getVisibleUnitType() == UnitType::Bernoulli)
                f = sigm(f);

              debugReconstructions->push_back(boost::make_shared<host_tensor_t>(v));
              debugReconstructions->push_back(boost::make_shared<host_tensor_t>(f));
              debugReconstructions->push_back(boost::make_shared<host_tensor_t>(vneg));
            }

            if (iSample == 0 && monitorTraining) {
              vnegmean = sum(vneg) / vneg.count();
            }
    #endif

            if (getCalculateError()) {
              batchError += sqrt(dot(vneg_master - v, vneg_master - v) / v.count());
            }
            cudaStreamSynchronize(0);
          }

          // Wait until master is done and copy result of cvneg_master to local thread copies
          #pragma omp barrier

          vneg = vneg_master;

          cudaStreamSynchronize(0);
          #pragma omp barrier

          for (size_t k = tid; k < F.size(); k += gpuCount) {

            // h_k = sigm(~F_k * v + c)
//            h_full = filter(vneg, flip(*F[k]), dimCount - 1);
            switch (getFilterMethod()) {
            case FilterMethod::FFT:
              h_full = filter(vneg, flip(*F[k]), dimCount - 1);
              break;
            case FilterMethod::NaiveConvolution:
              h_full = filter3d(vneg, *F[k], naive());
              break;
            case FilterMethod::OptimizedConvolution:
              h_full = filter3d(vneg, *F[k], optimized());
              break;
            case FilterMethod::ConvNet:
            case FilterMethod::NoConv:
              break;
            }
            if (getFilterMethod() != FilterMethod::ConvNet && getFilterMethod() != FilterMethod::NoConv)
              h = sum(h_full, dimCount - 1);
            else
              h.resize(layerSize, layerSize);
            h2 = h + *c[k];

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

            // dF_k = ~h * v
            switch (getFilterMethod()) {
            case FilterMethod::FFT:
              f = filter(vneg, repeat(flip(h), vneg.size() / h.size()), F[k]->size(), dimCount - 1);
              break;
            case FilterMethod::NaiveConvolution:
              h_full = filter3d(vneg, *F[k], naive());
              break;
            case FilterMethod::OptimizedConvolution:
              h_full = filter3d(vneg, *F[k], optimized());
              break;
            case FilterMethod::ConvNet:
            case FilterMethod::NoConv:
              break;
            }
            *Finc[k] = *Finc[k] - epsilonw * f;
            *cinc[k] = *cinc[k] - epsilonhb * h;
          }

          //binc = binc - epsilonvb * sum(vneg);
          #pragma omp master
          binc = binc - epsilonvb * vneg;
#endif

        } /* end of sample */

        for (size_t k = tid; k < F.size(); k += gpuCount) {
          *F[k] = *F[k] + *Finc[k];
          *c[k] = *c[k] + *cinc[k];
        }
        #pragma omp master
        {
          b = b + binc;

          if (monitor)
            monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));

          error += batchError;
        }

  #ifdef MONITOR_TRAINING
        if (monitorTraining) {

          #pragma omp master
          {
            f = ifft(cb, dimCount - 1, iplan_v);
            bmean = sum(f) / f.count();
          }
          value_t tc = 0, tF = 0;
          for (size_t k = tid; k < filters.size(); k += gpuCount) {
            l = ifft(*cc[k], dimCount - 1, iplan_h);
            tc += sum(l) / l.count();  // check what to do here with synchronization
            f = ifft(*cF[k], dimCount - 1, iplan_v);
            tF += sum(f) / f.count();
          }
          #pragma omp critical
          {
            cmean += tc;
            Fmean += tF;
          }
          #pragma omp barrier
          #pragma omp master
          {
            cmean /= filters.size();
            Fmean /= filters.size();

            f = ifft(cb, dimCount - 1, iplan_v);
            bsd = sqrt(dot(f - bmean, f - bmean) / f.count());
          }
          tc = tF = 0;
          for (size_t k = tid; k < filters.size(); k += gpuCount) {
            l = ifft(*cc[k], dimCount - 1, iplan_h);
            tc += dot(l - cmean, l - cmean) / l.count();
            f = ifft(*cF[k], dimCount - 1, iplan_v);
            tF += dot(f - Fmean, f - Fmean) / f.count();
          }
          #pragma omp critical
          {
            csd += tc;
            Fsd += tF;
          }
          #pragma omp barrier
          #pragma omp master
          {
            csd = sqrt(csd / filters.size());
            Fsd = sqrt(Fsd / filters.size());

            logfile << bmean << ", " << bsd << ", " << cmean << ", " << csd << ", " << Fmean << ", " << Fsd << ", " << hmean << ", " << vnegmean << ", " << batchError / batchSize << std::endl;
          }
        }
  #endif

      } /* end of batch */

  #ifdef MONITOR_TRAINING
      if (getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
        #pragma omp master
        {
          debugFilters->clear();
          debugVisibleBiases->clear();
          debugHiddenBiases->clear();
        }
        #pragma omp barrier
        for (size_t k = tid; k < filters.size(); k += gpuCount) {
          f = ifft(*cF[k], dimCount - 1, iplan_v);
          #pragma omp critical
          debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(f)));
          l = ifft(*cc[k], dimCount - 1, iplan_h);
          #pragma omp critical
          debugHiddenBiases->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(l)));
        }
        #pragma omp master
        {
          f = ifft(cb, dimCount - 1, iplan_v);
          debugVisibleBiases->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(f)));

          newState->setFilters(debugFilters);
          newState->setVisibleBiases(debugVisibleBiases);
          newState->setHiddenBiases(debugHiddenBiases);
          newState->setHiddenUnits(debugHiddenUnits);
          newState->setReconstructions(debugReconstructions);
        }
      }
  #endif

      #pragma omp master
      {
        if (getCalculateError())
          dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / tensors.size();
        else
          dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

        if (monitor)
          monitor->reportProgress(100. * (iEpoch + 1) / epochCount);
      }
    } /* end of epochs */

    #pragma omp master
    {
      newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
    }

  #ifdef MONITOR_TRAINING
    #pragma omp master
    if (monitorTraining)
      logfile.close();
  #endif

    {
      tensor_t f2;
      for (size_t k = tid; k < F.size(); k += gpuCount) {
//        dim_t topleft = (filters[k]->size() - crbm->getFilterKernelSize() + 1) / 2;
//        tensor_t f = zeros<value_t>(filters[k]->size());
//        f[topleft, crbm->getFilterKernelSize()] = *F[k];
//        f2 = ifftshift(f, dimCount - 1);
//        *filters[k] = f2; //*F[k] * (abs(*F[k]) > 1e-16);
        *filters[k] = *F[k];
        *hb[k] = *c[k] * (abs(*c[k]) > 1e-16);
      }
      #pragma omp master
      {
        vb = b * (abs(b) > 1e-16);
      }
    }

    // Free up memory
    for (size_t k = tid; k < F.size(); k += gpuCount) {
      F[k] = Finc[k] = c[k] = cinc[k] = boost::shared_ptr<tensor_t>();
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

#ifdef MONITOR_TRAINING
  {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
          new std::vector<boost::shared_ptr<host_tensor_t> >());
    for (size_t i = 0; i < filters.size(); ++i) {
      debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(*filters[i])));
    }
    newState->setFilters(debugFilters);
  }
#endif

  newState->setModel(crbm);
}

}

}
