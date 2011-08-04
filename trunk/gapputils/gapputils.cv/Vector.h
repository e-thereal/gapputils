/*
 * Vector.h
 *
 *  Created on: Aug 4, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_VECTOR_H_
#define GAPPUTILSCV_VECTOR_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace cv {

class Vector : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(Vector)

  Property(InputVector, std::vector<float>)
  Property(OutputVector, boost::shared_ptr<std::vector<float> >)

private:
  mutable Vector* data;

public:
  Vector();
  virtual ~Vector();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_VECTOR_H_ */
