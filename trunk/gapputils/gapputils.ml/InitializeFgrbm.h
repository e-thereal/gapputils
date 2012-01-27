/*
 * InitializeFgrbm.h
 *
 *  Created on: Jan 25, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_INITIALIZEFGRBM_H_
#define GAPPUTILS_ML_INITIALIZEFGRBM_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class InitializeFgrbm : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(InitializeFgrbm)                                       // 4
  Property(ConditionalsVector, boost::shared_ptr<std::vector<double> >)       // 12
  Property(VisiblesVector, boost::shared_ptr<std::vector<double> >)           // 20
  Property(VisibleCount, int)                                                 // 24
  Property(HiddenCount, int)                                                  // 38
  Property(FactorCount, int)                                                  // 32
  int i2;
  Property(WeightStddevs, double)                                             // 40
  Property(DiagonalWeightMeans, double)                                       // 48
  Property(InitialHidden, double)                                             // 56
  Property(IsGaussian, bool)                                                  // 57

  Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)                         // 68

private:
  mutable InitializeFgrbm* data;

public:
  InitializeFgrbm();
  virtual ~InitializeFgrbm();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILS_ML_INITIALIZEFGRBM_H_ */
