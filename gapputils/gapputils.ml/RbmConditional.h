/*
 * RbmConditional.h
 *
 *  Created on: Mar 12, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMCONDITIONAL_H_
#define GAPPUTILS_ML_RBMCONDITIONAL_H_

#include <gapputils/WorkflowElement.h>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

/*!
 * Visible vector = [givens conditionals]
 */
class RbmConditional : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmConditional)

  Property(Model, boost::shared_ptr<RbmModel>)
  Property(Givens, boost::shared_ptr<std::vector<float> >)
  Property(Conditionals, boost::shared_ptr<std::vector<float> >)
  Property(GivenCount, int)
  Property(InitializationCycles, int)
  Property(SampleCycles, int)
  Property(ShowSamples, bool)
  Property(Delay, int)
  Property(Debug, bool)

private:
  mutable RbmConditional* data;

public:
  RbmConditional();
  virtual ~RbmConditional();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RBMCONDITIONAL_H_ */
