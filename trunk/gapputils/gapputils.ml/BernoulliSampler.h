/*
 * BernoulliSampler.h
 *
 *  Created on: Oct 22, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_BERNOULLISAMPLER_H_
#define GAPPUTILS_ML_BERNOULLISAMPLER_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {
namespace ml {

class BernoulliSampler : public workflow::DefaultWorkflowElement<BernoulliSampler> {

  InitReflectableClass(BernoulliSampler)

  Property(Parameters, boost::shared_ptr<std::vector<float> >)
  Property(SampleCount, int)
  Property(Data, boost::shared_ptr<std::vector<float> >)

public:
  BernoulliSampler();

protected:
  void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */
} /* namespace gapputils */
#endif /* BERNOULLISAMPLER_H_ */
