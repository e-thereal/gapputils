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

  typedef Model::value_t value_t;
  typedef Model::matrix_t host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef matrix_t::dim_t dim_t;

  typedef boost::shared_ptr<std::vector<double> > data_t;

  friend class TrainerChecker;

  InitReflectableClass(Trainer)

  Property(TrainingSet, boost::shared_ptr<std::vector<data_t> >)
  Property(DbmLayer, DbmLayer)
  Property(Mask, data_t)
  Property(AutoCreateMask, bool)
  Property(HiddenCount, int)
  Property(SampleHiddens, bool)
  Property(EpochCount, int)
  Property(BatchSize, int)
  int dummy;
  Property(LearningRate, value_t)
  Property(InitialWeights, value_t)
  Property(InitialVisible, value_t)
  Property(InitialHidden, value_t)
  Property(SparsityTarget, value_t)
  Property(SparsityWeight, value_t)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)
  Property(ShowWeights, int)
  Property(ShowEvery, int)

  Property(Model, boost::shared_ptr<Model>)
  Property(Weights, boost::shared_ptr<host_matrix_t>)
  Property(DebugMask, data_t)
  Property(DebugMask2, data_t)

public:
  Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_TRAINER_H_ */
