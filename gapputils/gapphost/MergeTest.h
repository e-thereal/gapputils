/*
 * MergeTest.h
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#ifndef GAPPHOST_MERGETEST_H_
#define GAPPHOST_MERGETEST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gapputils {

namespace testing {

class MergeTest : public DefaultWorkflowElement<MergeTest> {

  InitReflectableClass(MergeTest)

  Property(Inputs, std::vector<double>)
  Property(Outputs, std::vector<double>)

public:
  MergeTest();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace testing */

} /* namespace gapputils */

#endif /* GAPPHOST_MERGETEST_H_ */
