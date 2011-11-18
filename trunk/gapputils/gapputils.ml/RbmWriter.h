/*
 * RbmWriter.h
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_RBMWRITER_H_
#define GAPPUTILS_ML_RBMWRITER_H_

#include <gapputils/WorkflowElement.h>

#include "RbmModel.h"

namespace gapputils {

namespace ml {

class RbmWriter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RbmWriter)

  Property(RbmModel, boost::shared_ptr<RbmModel>)
  Property(Filename, std::string)

private:
  mutable RbmWriter* data;

public:
  RbmWriter();
  virtual ~RbmWriter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_RBMWRITER_H_ */
