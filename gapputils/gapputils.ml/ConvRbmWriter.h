/*
 * RbmWriter.h
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_CONVRBMWRITER_H_
#define GAPPUTILS_ML_CONVRBMWRITER_H_

#include <gapputils/WorkflowElement.h>

#include "ConvRbmModel.h"

namespace gapputils {

namespace ml {

class ConvRbmWriter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(ConvRbmWriter)

  Property(Model, boost::shared_ptr<ConvRbmModel>)
  Property(Filename, std::string)
  Property(AutoSave, bool)

private:
  mutable ConvRbmWriter* data;
  static int inputId;

public:
  ConvRbmWriter();
  virtual ~ConvRbmWriter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_CONVRBMWRITER_H_ */
