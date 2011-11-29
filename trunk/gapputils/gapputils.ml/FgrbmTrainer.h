/*
 * FgrbmTrainer.h
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMTRAINER_H_
#define GAPPUTILS_ML_FGRBMTRAINER_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class FgrbmTrainer : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FgrbmTrainer)

    Property(ConditionalsVector, boost::shared_ptr<std::vector<float> >)
    Property(VisiblesVector, boost::shared_ptr<std::vector<float> >)
    Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)
    Property(VisibleCount, int)
    Property(HiddenCount, int)
    Property(FactorCount, int)
    Property(SampleHiddens, bool)
    Property(EpochCount, int)
    Property(BatchSize, int)
    Property(LearningRate, float)
    Property(InitialHidden, float)
    Property(IsGaussian, bool)

private:
  mutable FgrbmTrainer* data;

public:
  FgrbmTrainer();
  virtual ~FgrbmTrainer();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FGRBMTRAINER_H_ */
