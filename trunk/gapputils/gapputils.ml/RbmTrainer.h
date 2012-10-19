/*
 * RbmTrainer.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMTRAINER_H_
#define GAPPUTILS_ML_RBMTRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <boost/shared_ptr.hpp>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

class RbmTrainer : public workflow::DefaultWorkflowElement<RbmTrainer> {

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
  Property(Weights, boost::shared_ptr<std::vector<float> >)
  Property(ShowWeights, int)
  Property(ShowEvery, int)
  //Property(PosData, boost::shared_ptr<std::vector<float> >)
  //Property(NegData, boost::shared_ptr<std::vector<float> >)

public:
  RbmTrainer();
  virtual ~RbmTrainer();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}


#endif /* GAPPUTILS_ML_RBMTRAINER_H_ */
