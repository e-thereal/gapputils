/*
 * OnOfN.h
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_ONEOFN_H_
#define GAPPUTILS_ML_ONEOFN_H_

#include <gapputils/WorkflowElement.h>

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace ml {

class OneOfN : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(OneOfN)

  Property(OnIndex, int)
  Property(Count, int)
  Property(Vector, boost::shared_ptr<std::vector<float> >)

private:
  mutable OneOfN* data;

public:
  OneOfN();
  virtual ~OneOfN();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_ONEOFN_H_ */
