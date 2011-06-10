/*
 * PrincipleComponents.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#ifndef GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_
#define GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_

#include <gapputils/WorkflowElement.h>

namespace GaussianProcesses {

class PrincipleComponents : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(PrincipleComponents)

  Property(FeatureCount, int)
  Property(SampleCount, int)
  Property(Data, double*)
  Property(PrincipleComponents, double*)

private:
  mutable PrincipleComponents* data;

public:
  PrincipleComponents();
  virtual ~PrincipleComponents();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

void getPcs(double* pc, double* data, int m, int n);

}


#endif /* GAUSSIANPROCESSES_PRINCIPLECOMPONENTS_H_ */
