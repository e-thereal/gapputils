/*
 * Mean.h
 *
 *  Created on: Mar 13, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_MEAN_H_
#define GAPPUTLIS_MEAN_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class Mean : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Mean)

  Property(InputVectors, boost::shared_ptr<std::vector<float> >)
  Property(FeatureCount, int)
  Property(OutputVector, boost::shared_ptr<std::vector<float> >)

private:
  mutable Mean* data;

public:
  Mean();
  virtual ~Mean();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTLIS_MEAN_H_ */
