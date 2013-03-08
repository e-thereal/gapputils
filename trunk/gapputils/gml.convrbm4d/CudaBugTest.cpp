/*
 * CudaBugTest.cpp
 *
 *  Created on: Mar 8, 2013
 *      Author: tombr
 */

#include "CudaBugTest.h"

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(CudaBugTest)

  ReflectableBase(DefaultWorkflowElement<CudaBugTest>)

EndPropertyDefinitions

CudaBugTest::CudaBugTest() {
  setLabel("BugTest");
}

} /* namespace convrbm4d */

} /* namespace gml */
