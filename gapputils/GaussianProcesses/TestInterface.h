/*
 * TestInterface.h
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#ifndef TESTINTERFACE_H_
#define TESTINTERFACE_H_

#include <gapputils/WorkflowInterface.h>

namespace GaussianProcesses {

class TestInterface : public gapputils::workflow::WorkflowInterface
{
  InitReflectableClass(TestInterface)

  Property(Pdf, std::string)

public:
  TestInterface();
  virtual ~TestInterface();
};

}

#endif /* TESTINTERFACE_H_ */
