/*
 * Trainer_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

//#define BOOST_CHRONO_HEADER_ONLY

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

#include <boost/timer.hpp>
//#include <boost/chrono.hpp>

#include <fstream>
#include <omp.h>

#include "math.hpp"

// All monitoring code is enclosed by #ifdef MONITOR_TRAINING blocks
#define MONITOR_TRAINING

namespace gml {

namespace convrbm {

TrainerChecker::TrainerChecker() {
  Trainer trainer;
  trainer.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateW, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateVB, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRateHB, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
  CHECK_MEMORY_LAYOUT2(Logfile, trainer);
  CHECK_MEMORY_LAYOUT2(MonitorEvery, trainer);
  CHECK_MEMORY_LAYOUT2(ReconstructionCount, trainer);

  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(Filters, trainer);
  CHECK_MEMORY_LAYOUT2(VisibleBiases, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenBiases, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenUnits, trainer);
  CHECK_MEMORY_LAYOUT2(Reconstructions, trainer);
}

//class timer {
//private:
//  boost::chrono::process_real_cpu_clock clock;
//  boost::chrono::process_real_cpu_clock::time_point timePoint;
//
//public:
//  timer() {
//    timePoint = clock.now();
//  }
//
//  boost::chrono::duration<double> elapsed() {
//    return (clock.now() - timePoint);
//  }
//
//  void restart() {
//    timePoint = clock.now();
//  }
//};

#define START size_t timerCycles = getEpochCount(); \
    boost::timer _timer;

#define STOP { \
    cudaThreadSynchronize(); \
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

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace thrust::placeholders;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  const int gpuCount = getGpuCount();

  if (deviceCount < gpuCount) {
    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << gpuCount << " are required according to GpuCount. Aborting!";
    return;
  }

  assert(omp_get_num_threads() == 1);

  cudaSetDevice(0);
  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

  const unsigned dimCount = Model::dimCount;
  typedef complex<value_t> complex_t;
  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor<complex_t, dimCount, false> host_ctensor_t;
  typedef tensor_t::dim_t dim_t;

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

  std::vector<boost::shared_ptr<host_ctensor_t> > cX;

  // Normalize input and pre-calculate the FFT
  {
    tensor_t x;
    ctensor_t cx;
    plan_t plan_v;
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
      cx = fft(x, dimCount - 1, plan_v);
      cX.push_back(boost::shared_ptr<host_ctensor_t>(new host_ctensor_t(cx)));
    }
  }

  // Create arrays for filters
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<boost::shared_ptr<ctensor_t> > cF(filters.size()), cFinc(filters.size());

  // Copy visible bias to the device
  host_tensor_t& b = *crbm->getVisibleBias();
  ctensor_t cb, cbinc;
  cb.set_name("cb");
  cbinc.set_name("cbinc");
  {
    tensor_t f = b;
    plan_t plan_v;
    if (getShareBiasTerms())
      f = ones<value_t>(f.size()) * sum(f) / f.count();
    cb = fft(f, dimCount - 1, plan_v);
    cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
  }

  std::vector<boost::shared_ptr<host_tensor_t> >& c = *crbm->getHiddenBiases();
  std::vector<boost::shared_ptr<ctensor_t> > cc(c.size()), ccinc(c.size());

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
  value_t epsilonvb = getLearningRateVB() / batchSize;                  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRateHB() / batchSize;                  // Learning rate for biases of hidden units
  value_t epsilonsb = getSparsityWeight() / batchSize;                  // Sparsity weight
  value_t weightcost = 0.0002 * getLearningRateW();
  value_t initialmomentum = 0.5; //65; // 0.5f;
  value_t finalmomentum = 0.9; // 65; // 0.9f;
  value_t momentum;

  // Declare variables used for training
  tensor_t v, vneg; // to monitor training (calculate the error)
  ctensor_t cv_master;      // Read from the master thread and then each other thread reads from the master device
  ctensor_t cvneg_master;   // All threads add their version to the master threads version (needs to be synchronized)
  v.set_name("v");
  vneg.set_name("vneg");
  cv_master.set_name("cv_master");
  cvneg_master.set_name("cvneg_master");

  dim_t vbMaskSize = cb.size(), hbMaskSize, spMaskSize;
  if (getShareBiasTerms()) {
    vbMaskSize[0] = 1;
    vbMaskSize[1] = 1;
  }

  bool monitorTraining = false;
  value_t error = 0, batchError = 0;

  #pragma omp parallel
  {
    /*** PREPARE GPU THREADS ***/

    int tid = omp_get_thread_num();
    cudaSetDevice(tid);

    // Enable peer to peer access of each card with the master card and vice versa
    if (tid == 0) {
      for (int i = 1; i < gpuCount; ++i)
        cudaDeviceEnablePeerAccess(i, 0);
    } else {
      cudaDeviceEnablePeerAccess(0, 0);
    }
    #pragma omp barrier

#ifdef MONITOR_TRAINING
    tensor_t l;
    ctensor_t dcv;
#endif

    #pragma omp master
    assert(tid == 0);   // Check the assumption that the first thread is the master thread

    // FFT plans
    plan_t plan_h, iplan_h, plan_v, iplan_v;

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f;
      ctensor_t cf;
      for (size_t i = tid; i < filters.size(); i += gpuCount) {
        f = *filters[i];
        cf = fft(f, dimCount - 1, plan_v);
        cF[i] = boost::make_shared<ctensor_t>(cf);
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cf.size(), cf.fullsize()));
      }
    }
    {
      tensor_t h;
      ctensor_t ch;
      for (size_t i = tid; i < c.size(); i += gpuCount) {
        h = *c[i];
        if (getShareBiasTerms())
          h = ones<value_t>(h.size()) * sum(h) / h.count();
        ch = fft(h, dimCount - 1, plan_h);
        cc[i] = boost::make_shared<ctensor_t>(ch);
        ch = zeros<complex_t>(ch.size(), ch.fullsize());
        ccinc[i] = boost::make_shared<ctensor_t>(ch);
      }
    }

    // Declare variables used for training
    tensor_t h, h2, f;       // for sigm, sampling and masking
    ctensor_t ch, ch_full;
    ctensor_t cv;         // Read from the master thread and then each other thread reads from the master device
    ctensor_t cvneg;      // All threads add their version to the master threads version (needs to be synchronized)
    random_tensor<value_t, dimCount, true, uniform<value_t> > h_rand(layerSize, tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise(layerSize, tid);

    #pragma omp master
    {
      hbMaskSize = cc[0]->size();
      spMaskSize = cc[0]->size();
      if (getShareBiasTerms()) {
        hbMaskSize[0] = 1;
        hbMaskSize[1] = 1;
      }
      spMaskSize[0] = 1;
      spMaskSize[1] = 1;
    }

    #pragma omp barrier

    #pragma omp master
    dlog() << "Trainer initialized. Starting training.";

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

        for (size_t k = tid; k < cF.size(); k += gpuCount) {
          *cFinc[k] = momentum * *cFinc[k] - weightcost * *cF[k];
          *ccinc[k] = momentum * *ccinc[k];
        }

        #pragma omp master
        cbinc = momentum * cbinc;

        for (size_t iSample = 0; iSample < batchSize; ++iSample) {

          cudaStreamSynchronize(0);
          #pragma omp barrier

          // get v
          #pragma omp master
          {
            if (getRandomizeTraining())
              cv_master = *cX[rand() % cX.size()];
            else
              cv_master = *cX[iSample + iBatch * batchSize];
            cvneg_master = zeros<complex_t>(cv_master.size(), cv_master.fullsize());
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          cv = cv_master;
          cvneg = zeros<complex_t>(cv.size(), cv.fullsize());

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

          for (size_t k = tid; k < cF.size(); k += gpuCount) {

            /*** BEGIN OF POSITIVE PHASE ***/

            // h_k = sigm(~F_k * v + c)
            ch_full = conj(*cF[k]) * cv;
            ch = sum(ch_full, dimCount - 1);
            ch = ch + *cc[k];
            h2 = ifft(ch, iplan_h);

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
            ch = fft(h, plan_h);
            *cFinc[k] = *cFinc[k] + epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;
            *ccinc[k] = *ccinc[k] + epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
            *ccinc[k] = *ccinc[k] + epsilonsb * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (getSparsityTarget() * h.count() + -ch);

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

            ch = fft(h, plan_h);

            /*** BEGIN OF NEGATIVE PHASE ***/

            // dvneg = F * h
            cvneg = cvneg + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
          }

          // Add up local copies
          #pragma omp critical
          {
            cvneg_master = cvneg_master + cvneg;
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
            if (getCalculateError() || monitorTraining)
              v = ifft(cv, dimCount - 1, iplan_v);

            //binc = binc + epsilonvb * sum(v);
            cbinc = cbinc + epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

            /*** END OF NEGATIVE PHASE ***/

            cvneg_master = cvneg_master + cb;

            if (getCalculateError() || monitorTraining)
              vneg = ifft(cvneg_master, dimCount - 1, iplan_v);

            switch(crbm->getVisibleUnitType()) {
              case UnitType::Bernoulli:
                if (!getCalculateError())
                  vneg = ifft(cvneg_master, dimCount - 1, iplan_v);
                vneg = sigm(vneg);
                cvneg_master = fft(vneg, dimCount - 1, plan_v);
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
              batchError += sqrt(dot(vneg - v, vneg - v) / v.count());
            }
            cudaStreamSynchronize(0);
          }

          // Wait until master is done and copy result of cvneg_master to local thread copies
          #pragma omp barrier

          cvneg = cvneg_master;

          cudaStreamSynchronize(0);
          #pragma omp barrier

          for (size_t k = tid; k < cF.size(); k += gpuCount) {

            // h_k = sigm(~F_k * v + c)
            ch_full = conj(*cF[k]) * cvneg;
            ch = sum(ch_full, dimCount - 1);
            ch = ch + *cc[k];
            h2 = ifft(ch, iplan_h);

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
            ch = fft(h, plan_h);
            *cFinc[k] = *cFinc[k] - epsilonw * repeat(conj(ch), cvneg.size() / ch.size()) * cvneg;
            *ccinc[k] = *ccinc[k] - epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
          }

          //binc = binc - epsilonvb * sum(vneg);
          #pragma omp master
          cbinc = cbinc - epsilonvb * mask<complex_t>(cvneg.size(), cvneg.fullsize(), vbMaskSize) * cvneg;
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
          cb = cb + cbinc;

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
          dlog(Severity::Trace) << "Error: " << error / tensors.size();

        if (monitor)
          monitor->reportProgress(100. * (iEpoch + 1) / epochCount,  getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0);
      }
    } /* end of epochs */

  #ifdef MONITOR_TRAINING
    #pragma omp master
    if (monitorTraining)
      logfile.close();
  #endif

    {
      tensor_t hb;
      for (size_t k = tid; k < cF.size(); k += gpuCount) {
        f = ifft(*cF[k], dimCount - 1, iplan_v);
        hb = ifft(*cc[k], dimCount - 1, iplan_h);

        *filters[k] = f;
        *c[k] = hb;
      }
      #pragma omp master
      {
        f = ifft(cb, dimCount - 1, iplan_v);
        b = f;
      }
    }

    // Free up memory
    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      cF[k] = cFinc[k] = cc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
    }

    if (tid == 0) {
      for (int i = 1; i < gpuCount; ++i)
        cudaDeviceDisablePeerAccess(i);
    } else {
      cudaDeviceDisablePeerAccess(0);
    }
  } /* end of parallel code */

  {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
          new std::vector<boost::shared_ptr<host_tensor_t> >());
    for (size_t i = 0; i < filters.size(); ++i) {
      debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(*filters[i])));
    }
    newState->setFilters(debugFilters);
  }

  newState->setModel(crbm);
}

}

}
