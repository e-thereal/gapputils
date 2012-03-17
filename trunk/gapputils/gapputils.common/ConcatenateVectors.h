/*
 * ConcatenateVectors.h
 *
 *  Created on: Mar 12, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMMON_CONCATENATEVECTORS_H_
#define GAPPUTILS_COMMON_CONCATENATEVECTORS_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class ConcatenateVectors : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ConcatenateVectors)

  Property(Input1, boost::shared_ptr<std::vector<float> >)
  Property(Input2, boost::shared_ptr<std::vector<float> >)
  Property(Dimension1, int)
  Property(Dimension2, int)
  Property(Output, boost::shared_ptr<std::vector<float> >)
  Property(OutputDimension, int)

private:
  mutable ConcatenateVectors* data;

public:
  ConcatenateVectors();
  virtual ~ConcatenateVectors();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_COMMON_CONCATENATEVECTORS_H_ */
