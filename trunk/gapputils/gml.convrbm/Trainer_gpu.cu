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

#include <boost/timer.hpp>

#include <fstream>

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

  tensor_t f, l;
  ctensor_t dcv;
#endif

  // FFT plans
  plan_t plan_h, iplan_h, plan_v, iplan_v;

  boost::shared_ptr<Model> crbm = getInitialModel()->clone();

  std::vector<boost::shared_ptr<host_tensor_t> >& tensors = *getTensors();
  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  const size_t epochCount = getEpochCount();

  std::vector<boost::shared_ptr<host_ctensor_t> > cX;

  // Normalize input and pre-calculate the FFT
  {
    tensor_t x;
    ctensor_t cx, cv;
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

  // Copy filters to the device and pre-calculate the FFT
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm->getFilters();
  std::vector<ctensor_t > cF, cFinc;
  {
    tensor_t f;
    ctensor_t cf;
    for (size_t i = 0; i < filters.size(); ++i) {
      f = *filters[i];
      cf = fft(f, dimCount - 1, plan_v);
      cF.push_back(cf);
      cf = zeros<complex_t>(cf.size(), cf.fullsize());
      cFinc.push_back(cf);
    }
  }

  host_tensor_t& b = *crbm->getVisibleBias();
  ctensor_t cb, cbinc;
  {
    tensor_t f = b;
    if (getShareBiasTerms())
      f = ones<value_t>(f.size()) * sum(f) / f.count();
    cb = fft(f, dimCount - 1, plan_v);
    cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
  }

  std::vector<boost::shared_ptr<host_tensor_t> >& c = *crbm->getHiddenBiases();
  std::vector<ctensor_t> cc, ccinc;
  {
    tensor_t h;
    ctensor_t ch;
    for (size_t i = 0; i < c.size(); ++i) {
      h = *c[i];
      if (getShareBiasTerms())
        h = ones<value_t>(h.size()) * sum(h) / h.count();
      ch = fft(h, dimCount - 1, plan_h);
      cc.push_back(ch);
      ch = zeros<complex_t>(ch.size(), ch.fullsize());
      ccinc.push_back(ch);
    }
  }

  dlog() << "Trainer initialized.";

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
  tensor_t h, h2;       // for sigm and sampling
  ctensor_t cv, ch, ch_full, cvneg;
  random_tensor<value_t, dimCount, true, uniform<value_t> > h_rand(layerSize);
  random_tensor<value_t, dimCount, true, normal<value_t> > h_noise(layerSize);

  dim_t vbMaskSize = cb.size(), hbMaskSize = cc[0].size(), spMaskSize = cc[0].size();
  if (getShareBiasTerms()) {
    vbMaskSize[0] = hbMaskSize[0] = 1;
    vbMaskSize[1] = hbMaskSize[1] = 1;
  }
  spMaskSize[0] = 1;
  spMaskSize[1] = 1;

  dlog() << "Starting training";

  bool monitorTraining = false;
#ifdef MONITOR_TRAINING
  monitorTraining = getLogfile().size();
  std::ofstream logfile;
  value_t bmean, bsd, cmean, csd, Fmean, Fsd, hmean, vnegmean;

  if (monitorTraining) {
    logfile.open(getLogfile().c_str());

    logfile << "b.mean, b.sd, c.mean, c.sd, F.mean, F.sd, h.mean, vneg.mean, error" << std::endl;

    bmean = bsd = cmean = csd = Fmean = Fsd = hmean = vnegmean = 0;

    f = ifft(cb, dimCount - 1, iplan_v);
    bmean = sum(f) / f.count();
    for (size_t k = 0; k < filters.size(); ++k) {
      l = ifft(cc[k], dimCount - 1, iplan_h);
      cmean += sum(l) / l.count();
      f = ifft(cF[k], dimCount - 1, iplan_v);
      Fmean += sum(f) / f.count();
    }
    cmean /= filters.size();
    Fmean /= filters.size();

    f = ifft(cb, dimCount - 1, iplan_v);
    bsd = sqrt(dot(f - bmean, f - bmean) / f.count());
    for (size_t k = 0; k < filters.size(); ++k) {
      l = ifft(cc[k], dimCount - 1, iplan_h);
      csd += dot(l - cmean, l - cmean) / l.count();
      f = ifft(cF[k], dimCount - 1, iplan_v);
      Fsd += dot(f - Fmean, f - Fmean) / f.count();
    }
    csd = sqrt(csd / filters.size());
    Fsd = sqrt(Fsd / filters.size());

    logfile << bmean << ", " << bsd << ", " << cmean << ", " << csd << ", " << Fmean << ", " << Fsd << ", " << hmean << ", " << vnegmean << ", " << 0 << std::endl;
  }
#endif

  for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
    value_t error = 0;
    momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

#ifdef MONITOR_TRAINING
    debugHiddenUnits->clear();
    debugReconstructions->clear();
#endif

    for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {
      value_t batchError = 0;

#ifdef MONITOR_TRAINING
      bmean = bsd = cmean = csd = Fmean = Fsd = hmean = vnegmean = 0;
#endif

      for (size_t k = 0; k < cF.size(); ++k) {
        cFinc[k] = momentum * cFinc[k] - weightcost * cF[k];
        ccinc[k] = momentum * ccinc[k];

      }
      cbinc = momentum * cbinc;

      for (size_t iSample = 0; iSample < batchSize; ++iSample) {

        // get v
        if (getRandomizeTraining())
          cv = *cX[rand() % cX.size()];
        else
          cv = *cX[iSample + iBatch * batchSize];
        cvneg = zeros<complex_t>(cv.size(), cv.fullsize());

#ifdef MONITOR_TRAINING
        if (iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0)
          dcv = zeros<complex_t>(cv.size(), cv.fullsize());
#endif

        for (size_t k = 0; k < cF.size(); ++k) {

          /*** BEGIN OF POSITIVE PHASE ***/

          // h_k = sigm(~F_k * v + c)
          ch_full = conj(cF[k]) * cv;
          ch = sum(ch_full, dimCount - 1);
          ch = ch + cc[k];
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
          cFinc[k] = cFinc[k] + epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;

          ccinc[k] = ccinc[k] + epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch
              + epsilonsb * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (getSparsityTarget() * h.count() - sum(h));

#ifdef MONITOR_TRAINING
          if (iSample == 0 && iBatch == 0 && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
            debugHiddenUnits->push_back(boost::make_shared<host_tensor_t>(h));
          }

          if (iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
            dcv = dcv + cF[k] * repeat(ch, cF[k].size() / ch.size());
          }

          if (monitorTraining && iSample == 0)
            hmean += sum(h) / h.count() / cF.size();
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
            debugHiddenUnits->push_back(boost::make_shared<host_tensor_t>(h));
          }
#endif

          ch = fft(h, plan_h);

          /*** BEGIN OF NEGATIVE PHASE ***/

          // dvneg = F * h
          cvneg = cvneg + cF[k] * repeat(ch, cF[k].size() / ch.size());
        }

        /*** END OF POSITIVE PHASE ***/

        if (getCalculateError() || monitorTraining)
          v = ifft(cv, dimCount - 1, iplan_v);

        //binc = binc + epsilonvb * sum(v);
        cbinc = cbinc + epsilonvb * mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

        /*** END OF NEGATIVE PHASE ***/

        cvneg = cvneg + cb;

        if (getCalculateError() || monitorTraining)
          vneg = ifft(cvneg, dimCount - 1, iplan_v);

        switch(crbm->getVisibleUnitType()) {
          case UnitType::Bernoulli:
            if (!getCalculateError())
              vneg = ifft(cvneg, dimCount - 1, iplan_v);
            vneg = sigm(vneg);
            cvneg = fft(vneg, dimCount - 1, plan_v);
            break;

          case UnitType::Gaussian:
          break;

          default:
            dlog(Severity::Warning) << "Unsupported visible unit type: " << crbm->getVisibleUnitType();
        }

#ifdef MONITOR_TRAINING
        if (monitorTraining && iSample == 0 && iBatch < getReconstructionCount() && getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
          dcv = dcv + cb;
          f = ifft(dcv, dimCount - 1, iplan_v);
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
          if (batchError != batchError) {
            dlog(Severity::Error) << "An error occured during leraning";
            dlog(Severity::Error) << "sum(cb) = " << sum(cb);
            for (size_t k = 0; k < cF.size(); ++k) {
              dlog(Severity::Error) << "sum(cF[" << k << "]) = " << sum(cF[k]);
              dlog(Severity::Error) << "sum(cc[" << k << "]) = " << sum(cc[k]);
            }
            return;
          }
        }

        for (size_t k = 0; k < cF.size(); ++k) {

          // h_k = sigm(~F_k * v + c)
          ch_full = conj(cF[k]) * cvneg;
          ch = sum(ch_full, dimCount - 1);
          ch = ch + cc[k];
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
          cFinc[k] = cFinc[k] - epsilonw * repeat(conj(ch), cvneg.size() / ch.size()) * cvneg;

          ccinc[k] = ccinc[k] - epsilonhb * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
        }

        //binc = binc - epsilonvb * sum(vneg);
        cbinc = cbinc - epsilonvb * mask<complex_t>(cvneg.size(), cvneg.fullsize(), vbMaskSize) * cvneg;
      } /* end of sample */

      for (size_t k = 0; k < cF.size(); ++k) {
        f = ifft(cFinc[k], dimCount - 1, iplan_v);
        f = f * mask<value_t>(f.size(), crbm->getFilterKernelSize());
        cFinc[k] = fft(f, dimCount - 1, plan_v);
        cF[k] = cF[k] + cFinc[k];
        cc[k] = cc[k] + ccinc[k];
      }
      cb = cb + cbinc;

#ifdef MONITOR_TRAINING
      if (monitorTraining) {

        f = ifft(cb, dimCount - 1, iplan_v);
        bmean = sum(f) / f.count();
        for (size_t k = 0; k < filters.size(); ++k) {
          l = ifft(cc[k], dimCount - 1, iplan_h);
          cmean += sum(l) / l.count();
          f = ifft(cF[k], dimCount - 1, iplan_v);
          Fmean += sum(f) / f.count();
        }
        cmean /= filters.size();
        Fmean /= filters.size();

        f = ifft(cb, dimCount - 1, iplan_v);
        bsd = sqrt(dot(f - bmean, f - bmean) / f.count());
        for (size_t k = 0; k < filters.size(); ++k) {
          l = ifft(cc[k], dimCount - 1, iplan_h);
          csd += dot(l - cmean, l - cmean) / l.count();
          f = ifft(cF[k], dimCount - 1, iplan_v);
          Fsd += dot(f - Fmean, f - Fmean) / f.count();
        }
        csd = sqrt(csd / filters.size());
        Fsd = sqrt(Fsd / filters.size());

        logfile << bmean << ", " << bsd << ", " << cmean << ", " << csd << ", " << Fmean << ", " << Fsd << ", " << hmean << ", " << vnegmean << ", " << batchError / batchSize << std::endl;
      }
#endif

      if (monitor)
        monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));

      error += batchError;
    } /* end of batch */

    if (getCalculateError())
      dlog(Severity::Trace) << "Error: " << error / tensors.size();

#ifdef MONITOR_TRAINING
    if (getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0) {
      debugFilters->clear();
      debugVisibleBiases->clear();
      debugHiddenBiases->clear();
      for (size_t k = 0; k < filters.size(); ++k) {
        f = ifft(cF[k], dimCount - 1, iplan_v);
        debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(f)));
        l = ifft(cc[k], dimCount - 1, iplan_h);
        debugHiddenBiases->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(l)));
      }
      f = ifft(cb, dimCount - 1, iplan_v);
      debugVisibleBiases->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(f)));

      newState->setFilters(debugFilters);
      newState->setVisibleBiases(debugVisibleBiases);
      newState->setHiddenBiases(debugHiddenBiases);
      newState->setHiddenUnits(debugHiddenUnits);
      newState->setReconstructions(debugReconstructions);
    }
#endif

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount,  getMonitorEvery() > 0 && iEpoch % getMonitorEvery() == 0);
  }

#ifdef MONITOR_TRAINING
  if (monitorTraining)
    logfile.close();
#endif

  {
    tensor_t f, hb;
    for (size_t i = 0; i < cF.size(); ++i) {
      f = ifft(cF[i], dimCount - 1, iplan_v);
      *filters[i] = f;

      hb = ifft(cc[i], dimCount - 1, iplan_h);
      *c[i] = hb;
    }
    f = ifft(cb, dimCount - 1, iplan_v);
    b = f;
  }

  {
    boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > debugFilters(
          new std::vector<boost::shared_ptr<host_tensor_t> >());
    for (size_t i = 0; i < filters.size(); ++i) {
      debugFilters->push_back(boost::shared_ptr<host_tensor_t> (new host_tensor_t(*filters[i])));
    }
  }

  newState->setFilters(debugFilters);
  newState->setModel(crbm);
}

}

}
