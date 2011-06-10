/*
 * LinearTransformation.h
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#ifndef GAUSSIANPROCESSES_LINEARTRANSFORMATION_H_
#define GAUSSIANPROCESSES_LINEARTRANSFORMATION_H_

#include <gapputils/WorkflowElement.h>

namespace GaussianProcesses {

class LinearTransformation : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(LinearTransformation)

  Property(Transpose, bool)
  Property(OriginalFeatureCount, int)
  Property(ReducedFeatureCount, int)
  Property(SampleCount, int)
  Property(Transformation, double*)
  Property(Input, double*)
  Property(Output, double*)

private:
  mutable LinearTransformation* data;

public:
  LinearTransformation();
  virtual ~LinearTransformation();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

void lintrans(double* output, int outM, int n, double* input,
    int inM, double* transformation, bool transpose);

}


#endif /* GAUSSIANPROCESSES_LINEARTRANSFORMATION_H_ */
