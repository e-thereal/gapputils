/*
 * RbmTrainer.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMTRAINER_H_
#define GAPPUTILS_ML_RBMTRAINER_H_

#include <gapputils/WorkflowElement.h>
#include <boost/shared_ptr.hpp>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

class RbmTrainer : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmTrainer)

  Property(TrainingSet, boost::shared_ptr<std::vector<float> >)
  Property(RbmModel, boost::shared_ptr<RbmModel>)
  Property(VisibleCount, int)
  Property(HiddenCount, int)
  Property(SampleHiddens, bool)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(LearningRate, float)
  Property(InitialHidden, float)
  Property(SparsityTarget, float)
  Property(SparsityWeight, float)
  Property(IsGaussian, bool)
  //Property(PosData, boost::shared_ptr<std::vector<float> >)
  //Property(NegData, boost::shared_ptr<std::vector<float> >)

private:
  mutable RbmTrainer* data;

public:
  RbmTrainer();
  virtual ~RbmTrainer();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_ML_RBMTRAINER_H_ */
