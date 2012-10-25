/*
 * FreeEnergyClassifier.h
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FREEENERGYCLASSIFIER_H_
#define GAPPUTILS_ML_FREEENERGYCLASSIFIER_H_

#include <gapputils/DefaultWorkflowElement.h>

#include "RbmModel.h"

namespace gapputils {
namespace ml {

class FreeEnergyClassifier : public workflow::DefaultWorkflowElement<FreeEnergyClassifier> {

  typedef float value_t;

  InitReflectableClass(FreeEnergyClassifier)

  Property(Conditionals, boost::shared_ptr<std::vector<value_t> >)
  Property(Rbm, boost::shared_ptr<RbmModel>)
  Property(FeatureCount, int)
  Property(MakeBernoulli, bool)
  Property(Differences, boost::shared_ptr<std::vector<value_t> >)

public:
  FreeEnergyClassifier();
  virtual ~FreeEnergyClassifier();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_FREEENERGYCLASSIFIER_H_ */
