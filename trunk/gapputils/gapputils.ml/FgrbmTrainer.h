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

  Property(ConditionalsVector, boost::shared_ptr<std::vector<double> >)
  Property(VisiblesVector, boost::shared_ptr<std::vector<double> >)
  Property(InitialFgrbmModel, boost::shared_ptr<FgrbmModel>)
  Property(SampleVisibles, bool)
  Property(EpochCount, int)
  Property(BatchSize, int)
  int i1;
  Property(LearningRate, double)
  Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)

  Property(Wx, boost::shared_ptr<std::vector<float> >)
  Property(Wy, boost::shared_ptr<std::vector<float> >)

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
