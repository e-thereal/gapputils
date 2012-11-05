/*
 * RbmReader.h
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMREADER_H_
#define GAPPUTILS_ML_RBMREADER_H_

#include <gapputils/WorkflowElement.h>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

class RbmReader : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmReader)

  Property(Filename, std::string)
  Property(RbmModel, boost::shared_ptr<RbmModel>)
  Property(VisibleCount, int)
  Property(HiddenCount, int)
  Property(HiddenUnitType, HiddenUnitType)

private:
  mutable RbmReader* data;

public:
  RbmReader();
  virtual ~RbmReader();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RBMREADER_H_ */
