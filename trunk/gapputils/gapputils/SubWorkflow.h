#pragma once
#ifndef _GAPPUTILS_SUBWORKFLOW_H_
#define _GAPPUTILS_SUBWORKFLOW_H_

#include <gapputils/WorkflowInterface.h>

namespace interfaces {

class SubWorkflow : public gapputils::workflow::WorkflowInterface {
  InitReflectableClass(SubWorkflow)

  Property(Atomic, bool)

public:
  SubWorkflow(void);
};

}

#endif
