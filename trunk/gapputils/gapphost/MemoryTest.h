/*
 * MemoryTest.h
 *
 *  Created on: Apr 5, 2013
 *      Author: tombr
 */

#ifndef GML_MEMORYTEST_H_
#define GML_MEMORYTEST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace debug {

class MemoryTest : public DefaultWorkflowElement<MemoryTest> {

  InitReflectableClass(MemoryTest)

  Property(Input, boost::shared_ptr<std::vector<double> >)
  Property(Size, int)
  Property(Iterations, int)
  Property(Delay, int)
  Property(Output, boost::shared_ptr<std::vector<double> >)

public:
  MemoryTest();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace debug */

#endif /* GML_MEMORYTEST_H_ */
