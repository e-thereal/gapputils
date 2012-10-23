/*
 * BernoulliTrainer.h
 *
 *  Created on: Oct 22, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_BERNOULLITRAINER_H_
#define GAPPUTILS_ML_BERNOULLITRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {
namespace ml {

class BernoulliTrainer : public workflow::DefaultWorkflowElement<BernoulliTrainer> {

  InitReflectableClass(BernoulliTrainer)

  Property(TrainingSet, boost::shared_ptr<std::vector<float> >)
  Property(FeatureCount, int)
  Property(Parameters, boost::shared_ptr<std::vector<float> >)

public:
  BernoulliTrainer();
  virtual ~BernoulliTrainer();

protected:
  void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_ML_BERNOULLITRAINER_H_ */
