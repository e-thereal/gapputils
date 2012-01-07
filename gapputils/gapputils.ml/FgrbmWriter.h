/*
 * FgrbmWriter.h
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMWRITER_H_
#define GAPPUTILS_ML_FGRBMWRITER_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class FgrbmWriter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FgrbmWriter)

  Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)
  Property(Filename, std::string)

private:
  mutable FgrbmWriter* data;

public:
  FgrbmWriter();
  virtual ~FgrbmWriter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FGRBMWRITER_H_ */
