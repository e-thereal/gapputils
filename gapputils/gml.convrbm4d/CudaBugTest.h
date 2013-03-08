/*
 * CudaBugTest.h
 *
 *  Created on: Mar 8, 2013
 *      Author: tombr
 */

#ifndef GML_CUDABUGTEST_H_
#define GML_CUDABUGTEST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace convrbm4d {

class CudaBugTest : public DefaultWorkflowElement<CudaBugTest> {

  InitReflectableClass(CudaBugTest)

public:
  CudaBugTest();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CUDABUGTEST_H_ */
