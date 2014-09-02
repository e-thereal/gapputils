/*
 * DummyLoop.h
 *
 *  Created on: Sep 2, 2014
 *      Author: tombr
 */

#ifndef GML_DUMMYLOOP_H_
#define GML_DUMMYLOOP_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace debug {

class DummyLoop : public DefaultWorkflowElement<DummyLoop> {

  InitReflectableClass(DummyLoop)

  Property(Iterations, int)
  Property(ShowProgress, bool)

public:
  DummyLoop();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace debug */

#endif /* GML_DUMMYLOOP_H_ */
