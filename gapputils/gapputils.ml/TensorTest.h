/*
 * TensorTest.h
 *
 *  Created on: Mar 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TENSORTEST_H_
#define GAPPUTILS_ML_TENSORTEST_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace ml {

namespace tensors {

namespace test {

class TensorTest : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(TensorTest)

private:
  mutable TensorTest* data;

public:
  TensorTest();
  virtual ~TensorTest();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

}

}

#endif /* GAPPUTILS_ML_TENSORTEST_H_ */
