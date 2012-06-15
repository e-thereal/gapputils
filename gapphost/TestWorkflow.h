/*
 * TestWorkflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef _GAPPUTILS_TESTWORKFLOW_H_
#define _GAPPUTILS_TESTWORKFLOW_H_

#include "Workflow.h"

#include <capputils/ObservableClass.h>

namespace gapputils {

class TestWorkflow : public capputils::reflection::ReflectableClass,
                     public capputils::ObservableClass
{
  InitReflectableClass(TestWorkflow)

  Property(Name, std::string)
  Property(In1, std::string)
  Property(Out1, std::string)

public:
  TestWorkflow();
  virtual ~TestWorkflow();
};

}

#endif /* TESTWORKFLOW_H_ */
