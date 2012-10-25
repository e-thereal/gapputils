/*
 * GenerateVectors.h
 *
 *  Created on: Oct 23, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ML_GENERATEVECTORS_H_
#define GAPPUTLIS_ML_GENERATEVECTORS_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {
namespace ml {

class GenerateVectors : public workflow::DefaultWorkflowElement<GenerateVectors> {

  InitReflectableClass(GenerateVectors)

  Property(Vectors, boost::shared_ptr<std::vector<float> >)
  Property(From, std::vector<float>)
  Property(StepCount, std::vector<int>)
  Property(To, std::vector<float>)

public:
  GenerateVectors();
  virtual ~GenerateVectors();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GENERATEVECTORS_H_ */
