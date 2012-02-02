/*
 * ResampleFgrbmModel.h
 *
 *  Created on: Feb 1, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RESAMPLEFGRBMMODEL_H_
#define GAPPUTILS_ML_RESAMPLEFGRBMMODEL_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class ResampleFgrbmModel : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ResampleFgrbmModel)

  Property(InputModel, boost::shared_ptr<FgrbmModel>)
  Property(InputWidth, int)
  Property(InputHeight, int)
  Property(OutputWidth, int)
  Property(OutputHeight, int)
  Property(OutputModel, boost::shared_ptr<FgrbmModel>)

private:
  mutable ResampleFgrbmModel* data;

public:
  ResampleFgrbmModel();
  virtual ~ResampleFgrbmModel();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RESAMPLEFGRBMMODEL_H_ */
