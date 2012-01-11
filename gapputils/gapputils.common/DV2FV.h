/*
 * DV2FV.h
 *
 *  Created on: Jan 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMMON_DV2FV_H_
#define GAPPUTILS_COMMON_DV2FV_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class DV2FV : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(DV2FV)

  Property(Input, boost::shared_ptr<std::vector<double> >)
  Property(Output, boost::shared_ptr<std::vector<float> >)

private:
  mutable DV2FV* data;

public:
  DV2FV();
  virtual ~DV2FV();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_COMMON_DV2FV_H_ */
