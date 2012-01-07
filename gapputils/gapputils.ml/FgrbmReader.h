/*
 * FgrbmReader.h
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMREADER_H_
#define GAPPUTILS_ML_FGRBMREADER_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class FgrbmReader : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FgrbmReader)

  Property(Filename, std::string)
  Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)
  Property(VisibleCount, int)
  Property(HiddenCount, int)

private:
  mutable FgrbmReader* data;

public:
  FgrbmReader();
  virtual ~FgrbmReader();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FGRBMREADER_H_ */
