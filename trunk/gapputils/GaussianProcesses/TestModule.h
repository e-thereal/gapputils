/*
 * TestModule.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef TESTMODULE_H_
#define TESTMODULE_H_

#include <WorkflowElement.h>

namespace GaussianProcesses {

class TestModule : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(TestModule)

  Property(X1, int)
  Property(X2, int)
  Property(Y1, int)
  Property(Y2, int)

private:
  bool upToDate;
  mutable int result;

public:
  TestModule();
  virtual ~TestModule();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif /* TESTMODULE_H_ */
