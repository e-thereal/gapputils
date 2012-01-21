/*
 * FV2DV.h
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMMON_FV2DV_H_
#define GAPPUTILS_COMMON_FV2DV_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class FV2DV : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FV2DV)

  Property(Input, boost::shared_ptr<std::vector<float> >)
  Property(Output, boost::shared_ptr<std::vector<double> >)

private:
  mutable FV2DV* data;

public:
  FV2DV();
  virtual ~FV2DV();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_COMMON_FV2DV_H_ */
