/*
 * TestModule.h
 *
 *  Created on: 2012-10-20
 *      Author: tombr
 */

#ifndef GAPPUTILS_TESTING_TESTMODULE_H_
#define GAPPUTILS_TESTING_TESTMODULE_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {
namespace testing {

class TestModule : public workflow::DefaultWorkflowElement<TestModule> {

  InitReflectableClass(TestModule)

  Property(Input, std::string)
  Property(Output, std::string)
  Property(Cycles, int)
  Property(Delay, int)

public:
	TestModule();
	virtual ~TestModule();

protected:
	void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace testing */
} /* namespace gapputils */
#endif /* GAPPUTILS_TESTING_TESTMODULE_H_ */
