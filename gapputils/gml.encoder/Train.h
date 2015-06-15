/*
 * Train.h
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#ifndef GML_TRAIN_H_
#define GML_TRAIN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include <capputils/Enumerators.h>

#include "Model.h"

#include <tbblas/deeplearn/objective_function.hpp>

namespace gml {

namespace encoder {

CapputilsEnumerator(TrainingMethod, ClassicMomentum, NesterovMomentum, AdaGrad, AdaDelta, Adam, RmsProp, HessianFree);

class OptimizationParameters : public capputils::reflection::ReflectableClass,
                               public ObservableClass
{
  InitReflectableClass(OptimizationParameters)
};

class MomentumParameters : public OptimizationParameters {
  InitReflectableClass(MomentumParameters)

  Property(LearningRate, float)
  Property(LearningDecayEpochs, int)
  Property(InitialMomentum, float)
  Property(FinalMomentum, float)
  Property(MomentumDecayEpochs, int)

public:
  MomentumParameters();

  float getLearningRate(int epoch) const;
  float getMomentum(int epoch) const;
};

class AdaGradParameters : public OptimizationParameters {
  InitReflectableClass(AdaGradParameters)

  Property(LearningRate, float)
  Property(LearningDecayEpochs, int)
  Property(Epsilon, float)

public:
  AdaGradParameters();

  float getLearningRate(int epoch) const;
};

class AdaDeltaParameters : public OptimizationParameters {
  InitReflectableClass(AdaDeltaParameters)

  Property(Epsilon, float)
  Property(DecayRate, float)

public:
  AdaDeltaParameters();
};

class AdamParameters : public OptimizationParameters {
  InitReflectableClass(AdamParameters)

  Property(Alpha, float)
  Property(Beta1, float)
  Property(Beta2, float)
  Property(Epsilon, float)

public:
  AdamParameters();
};

class HessianFreeParameters : public OptimizationParameters {
  InitReflectableClass(HessianFreeParameters)

  Property(ConjugateGradientIterations, int)
  Property(InitialLambda, float)
  Property(Zeta, float)

public:
  HessianFreeParameters();
};

struct TrainChecker { TrainChecker(); } ;

class Train : public DefaultWorkflowElement<Train> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  friend class TrainChecker;

  InitReflectableClass(Train)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(EpochCount, int)
  Property(TrialEpochCount, int)
  Property(BatchSize, int)
  Property(FilterBatchSize, std::vector<int>)
  Property(SubRegionCount, host_tensor_t::dim_t)
  Property(Objective, tbblas::deeplearn::objective_function)
  Property(SensitivityRatio, double)
  Property(SharedBiasTerms, bool)

  Property(Method, TrainingMethod)
  Property(Parameters, boost::shared_ptr<OptimizationParameters>)
  Property(LearningRates, std::vector<double>)
  Property(LearningDecay, int)
  Property(InitialMomentum, double)
  Property(FinalMomentum, double)
  Property(MomentumDecayEpochs, int)
  Property(WeightCosts, double)
  Property(InitialWeights, std::vector<double>)
  Property(RandomizeTraining, bool)

  Property(AugmentedChannels, std::vector<int>)
  Property(ContrastSd, double)
  Property(BrightnessSd, double)
  Property(GammaSd, double)

  Property(SaveEvery, int)

  Property(CurrentEpoch, int)
  Property(Model, boost::shared_ptr<model_t>)
  Property(AugmentedSet, boost::shared_ptr<v_host_tensor_t>)

  static int methodId;

public:
  Train();

protected:
  virtual void update(IProgressMonitor* monitor) const;

  void changedHandler(ObservableClass* sender, int eventId);
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_TRAIN_H_ */
