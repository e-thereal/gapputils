/*
 * ConvRbmTrainer.h
 *
 *  Created on: Mar 5, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMTRAINER_H_
#define GAPPUTILS_ML_CONVRBMTRAINER_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

#include <tbblas/device_tensor.hpp>

namespace gapputils {

namespace ml {

class ConvRbmTrainer : public gapputils::workflow::WorkflowElement {
public:
  typedef ConvRbmModel::tensor_t host_tensor_t;
  typedef ConvRbmModel::value_t value_t;
  typedef tbblas::device_tensor<value_t, 3> device_tensor_t;

  InitReflectableClass(ConvRbmTrainer)

  Property(InitialModel, boost::shared_ptr<ConvRbmModel>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(SampleVisibles, bool)
  Property(EpochCount, int)
  Property(BatchSize, int)
  int i1;
  Property(LearningRate, value_t)
  Property(SparsityTarget, value_t)
  Property(SparsityPenalty, value_t)
  Property(UseRandomSamples, bool)
  Property(CalculateBaseline, bool)

  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(Filters, boost::shared_ptr<std::vector<float> >)

private:
  mutable ConvRbmTrainer* data;

public:
  ConvRbmTrainer();
  virtual ~ConvRbmTrainer();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMTRAINER_H_ */
