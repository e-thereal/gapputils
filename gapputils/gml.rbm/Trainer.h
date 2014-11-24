/*
 * Trainer.h
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#ifndef GML_TRAINER_H_
#define GML_TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace rbm {

CapputilsEnumerator(DbmLayer, RBM, VisibleLayer, IntermediateLayer, TopLayer);

struct TrainerChecker { TrainerChecker(); };

class Trainer : public DefaultWorkflowElement<Trainer> {

  typedef model_t::value_t value_t;
  typedef model_t::host_matrix_t host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef matrix_t::dim_t dim_t;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  friend class TrainerChecker;

  InitReflectableClass(Trainer)

  Property(TrainingSet, boost::shared_ptr<v_data_t>)
  Property(DbmLayer, DbmLayer)
  Property(Mask, boost::shared_ptr<data_t>)
  Property(AutoCreateMask, bool)
  Property(HiddenCount, int)
  Property(SampleHiddens, bool)
  Property(EpochCount, int)
  Property(BatchSize, int)
  int dummy;
  Property(LearningRate, double)
  Property(BiasLearningRate, double)
  Property(LearningDecay, int)
  Property(WeightDecay, double)
  Property(InitialWeights, double)
  Property(InitialVisible, double)
  Property(InitialHidden, double)
  Property(HiddenDropout, double)
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)
  Property(NormalizeIndividualUnits, bool)
  Property(ShuffleTrainingSet, bool)
  Property(ShowWeights, int)
  Property(ShowEvery, int)

  Property(FindLearningRate, bool)
  Property(TrialLearningRates, std::vector<double>)
  Property(TrialEpochCount, int)

  Property(Model, boost::shared_ptr<model_t>)

public:
  Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_TRAINER_H_ */
