/*
 * Trainer_gpu.cu
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#include "Trainer.h"

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

TrainerChecker::TrainerChecker() {
  Trainer test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(DbmLayer, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
  CHECK_MEMORY_LAYOUT2(AutoCreateMask, test);
  CHECK_MEMORY_LAYOUT2(HiddenCount, test);
  CHECK_MEMORY_LAYOUT2(SampleHiddens, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(LearningRates, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(WeightDecay, test);
  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
  CHECK_MEMORY_LAYOUT2(InitialVisible, test);
  CHECK_MEMORY_LAYOUT2(InitialHidden, test);
  CHECK_MEMORY_LAYOUT2(HiddenDropout, test)
  CHECK_MEMORY_LAYOUT2(SparsityTarget, test);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, test);
  CHECK_MEMORY_LAYOUT2(VisibleUnitType, test);
  CHECK_MEMORY_LAYOUT2(HiddenUnitType, test);
  CHECK_MEMORY_LAYOUT2(NormalizeIndividualUnits, test);
  CHECK_MEMORY_LAYOUT2(ShowWeights, test);
  CHECK_MEMORY_LAYOUT2(ShowEvery, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
}

//#define TIC timer.restart();
//#define TOC cudaThreadSynchronize(); std::cout << __LINE__ << ": " << timer.elapsed() << "s" << std::endl;
//#define REPEAT for(int i = 0; i < 1000; ++i)
#define TIC
#define TOC
#define REPEAT

#define TRACE std::cout << __LINE__ << std::endl;

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef random_tensor<value_t, 2, true, normal<value_t> > randn_t;

  boost::timer timer;
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  unit_type visibleUnitType = getVisibleUnitType();
  unit_type hiddenUnitType = getHiddenUnitType();

  dlog(Severity::Message) << "Building RBM with " << getHiddenUnitType() << " hidden units.";

  v_data_t& data = *getTrainingSet();

  // Calculate the mean and the std of all features
  const size_t visibleCount = data[0]->size();
  const size_t hiddenCount = getHiddenCount();
  const size_t sampleCount = data.size();
  const value_t sparsityTarget = getSparsityTarget();
  const value_t sparsityWeight = getSparsityWeight();

  matrix_t X(sampleCount, visibleCount);
  host_matrix_t h_X(sampleCount, visibleCount);
  for (size_t i = 0; i < sampleCount; ++i) {
    thrust::copy(data[i]->begin(), data[i]->end(), row(h_X, i).begin());
  }
  X = h_X;

  matrix_t mask(1, visibleCount);
  if (getMask()) {
    assert(getMask()->size() == visibleCount);
    thrust::copy(getMask()->begin(), getMask()->end(), mask.begin());
  } else if (getAutoCreateMask()) {
    mask = sum(X, 0);
    mask = mask > 1e-9;
  } else {
    mask = ones<value_t>(1, visibleCount);
  }

  host_matrix_t visibleMeans = zeros<value_t>(1, visibleCount);
  host_matrix_t visibleStds = ones<value_t>(1, visibleCount);

  if (visibleUnitType == unit_type::Gaussian) {
    matrix_t means;
    if (getNormalizeIndividualUnits()) {
      means = sum(X, 0);
      means = means / X.size()[0];
    } else {
      means = ones<value_t>(visibleMeans.size()) * sum(X) / X.count();
    }
    X = X - repeat(means, X.size() / means.size());

    matrix_t stddev;
    if (getNormalizeIndividualUnits()) {
      matrix_t temp = X * X;
      stddev = sum(temp, 0);
      stddev = sqrt(stddev / X.size()[0]) + (stddev == 0);  // If stddev == 0 set stddev to 1
    } else {
      stddev = ones<value_t>(visibleStds.size()) * sqrt(dot(X, X) / X.count());
    }
    X = X / repeat(stddev, X.size() / stddev.size()) * repeat(mask, X.size() / mask.size());

    visibleMeans = means;
    visibleStds = stddev;
  }

  if (getShuffleTrainingSet()){
    matrix_t trow;
    for (unsigned i = X.size()[0] - 1; i > 0; --i) {
      unsigned j = rand() % (i + 1);
      trow = row(X, i);
      row(X, i) = row(X, j);
      row(X, j) = trow;
    }
  }
  dlog() << "Rows shuffled: " << timer.elapsed() << " s";

  value_t epsilonw, epsilonvb, epsilonhb, initialWeight;

  std::vector<double> learningRates = getLearningRates();
  std::vector<double> initialWeights = getInitialWeights();
  value_t bestEpsilon, bestWeight, bestError;

  int epochCount = getEpochCount();

  for (int iWeight = 0; iWeight < initialWeights.size(); ++iWeight) {
    for (int iLearningRate = 0; iLearningRate < learningRates.size() + 1; ++iLearningRate) {

      // iLearningRate == learningRate.size() marks the final run, this is only done when all the weights were tried
      if (iLearningRate == learningRates.size() && iWeight < initialWeights.size() - 1)
        continue;

      // if only one weight and one learning rate is given, do the final run immediately
      if (iWeight == 0 && iLearningRate == 0 && initialWeights.size() == 1 && learningRates.size() == 1) {
        bestEpsilon = learningRates[0];
        bestWeight = initialWeights[0];
        continue;
      }

      if (iLearningRate < learningRates.size()) {
        epsilonhb = epsilonvb = epsilonw = learningRates[iLearningRate];
        initialWeight = initialWeights[iWeight];
        epochCount = getTrialEpochCount();
        dlog(Severity::Message) << "Trying learning rate of " << epsilonw << " with weight " << initialWeight;
      } else {
        epsilonhb = epsilonvb = epsilonw = bestEpsilon;
        initialWeight = bestWeight;
        dlog(Severity::Message) << "Final run with learning rate " << bestEpsilon << " and initial weight " << initialWeight;
        epochCount = getEpochCount();
      }

      boost::shared_ptr<model_t> model(new model_t());
      model->set_visibles_type(visibleUnitType);
      model->set_hiddens_type(hiddenUnitType);

      model->set_mean(visibleMeans);
      model->set_stddev(visibleStds);
      model->set_mask(mask);

      // Train the RBM
      // Initialize weights and bias terms

      // W_ij ~ N(mu = 0, sigma^2 = 0.1^2)
      matrix_t W = initialWeight * randn_t(visibleCount, hiddenCount);
      matrix_t b = getInitialVisible() * ones<value_t>(1, visibleCount);
      matrix_t c = getInitialHidden() * ones<value_t>(1, hiddenCount);

      W = W * repeat(trans(mask), W.size() / trans(mask).size());

      model->set_weights(W);
      model->set_visible_bias(b);
      model->set_hidden_bias(c);

      dlog() << "RBM initialized: " << timer.elapsed() << " s";

      // Start the learning
      const int batchSize = getBatchSize();
      const int batchCount = sampleCount / batchSize;
      float weightcost = getWeightDecay();
      float initialmomentum = 0.5f;
      float finalmomentum = 0.9f;
      float momentum;

      tbblas::deeplearn::rbm<value_t> rbm(*model);
      rbm.visibles().resize(seq((int)batchSize, (int)visibleCount));

      dlog() << "Preparation finished after " << timer.elapsed() << " s";
      dlog() << "Starting training";
      timer.restart();

      value_t error = 0, learningDecay = 1;

      for (int iEpoch = 0; iEpoch < epochCount && error == error && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

        error = 0;

        // Learning decay only during final learning
        if (getLearningDecay() > 1 && iLearningRate == learningRates.size()) {
          learningDecay = (value_t)getLearningDecay() / ((value_t)getLearningDecay() + (value_t)iEpoch);
        }

        if (iEpoch < 10)
          momentum = initialmomentum;
        else
          momentum = finalmomentum;

        for (int iBatch = 0; iBatch < batchCount && error == error && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

          /*** START POSITIVE PHASE ***/

          // Get current batch
          rbm.visibles() = X[seq(iBatch * batchSize, 0), rbm.visibles().size()];
          rbm.init_dropout(getHiddenDropout());
          rbm.init_gradient_updates(epsilonw, momentum, weightcost);

          rbm.infer_hiddens();
          rbm.update_positive_gradient(epsilonw * learningDecay, epsilonvb * learningDecay, epsilonhb * learningDecay);

          /*** END OF POSITIVE PHASE ***/

          rbm.sample_hiddens();
          rbm.sample_visibles();
          rbm.infer_hiddens();
          rbm.update_negative_gradient(epsilonw * learningDecay, epsilonvb * learningDecay, epsilonhb * learningDecay);

          /*** END OF NEGATIVE PHASE ***/

          error += sqrt(dot(rbm.visibles() - X[seq(iBatch * batchSize, 0), rbm.visibles().size()], rbm.visibles() - X[seq(iBatch * batchSize, 0), rbm.visibles().size()]) / rbm.visibles().count());
          momentum = (iEpoch > 5 ? finalmomentum : initialmomentum);

          rbm.apply_gradient();

          /*** END OF UPDATES ***/
        }
        int eta = timer.elapsed() / (iEpoch + 1) * (epochCount - iEpoch - 1);
        int sec = eta % 60;
        int minutes = (eta / 60) % 60;
        int hours = eta / 3600;
        dlog() << "Epoch " << iEpoch << " error " << (error / batchCount) << " after " << timer.elapsed() << "s. ETA: "
            << hours << " h " << minutes << " min " << sec << " s";

        if (monitor) {
          const int totalEpochs = getTrialEpochCount() * initialWeights.size() * learningRates.size() + getEpochCount();
          const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount();
          monitor->reportProgress(100 * (currentEpoch + 1) / totalEpochs,  getShowWeights() && (iEpoch % getShowEvery() == 0));
        }
      }

    //  if (getDbmLayer() == DbmLayer::IntermediateLayer) {
    //    rbm->setWeightMatrix(boost::make_shared<host_matrix_t>(0.5 * W));
    //  } else {
    //    rbm->setWeightMatrix(boost::make_shared<host_matrix_t>(W));
    //  }
    //  rbm->setVisibleBiases(boost::make_shared<host_matrix_t>(b));
    //  rbm->setHiddenBiases(boost::make_shared<host_matrix_t>(c));

      if (iLearningRate < learningRates.size()) {
        if ((iLearningRate == 0 && iWeight == 0) || !(error > bestError)) {   // using not greater instead of lesser to handle nan case.
          bestError = error;
          bestEpsilon = epsilonw;
          bestWeight = initialWeight;
          dlog(Severity::Message) << "Found better learning rate: " << epsilonw << " and initial weight: " << initialWeight << " with an error of " << bestError / batchCount << ".";
        }
      } else {
        newState->setModel(model);
      }
    }
  }
}

}

}
