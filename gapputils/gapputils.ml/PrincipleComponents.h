/*
 * PrincipleComponents.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#ifndef GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_
#define GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

class PrincipleComponents : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(PrincipleComponents)

  Property(Data, boost::shared_ptr<std::vector<float> >)
  Property(FeatureCount, int)

  Property(PrincipleComponents, boost::shared_ptr<std::vector<float> >)

private:
  mutable PrincipleComponents* data;

public:
  PrincipleComponents();
  virtual ~PrincipleComponents();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

void getPcs(float* pc, float* data, int m, int n);

}

}

#endif /* GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_ */
