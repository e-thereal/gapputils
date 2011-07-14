/*
 * AamWriter.h
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMWRITER_H_
#define GAPPUTILSCV_AAMWRITER_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamWriter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamWriter)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(Filename, std::string)

private:
  mutable AamWriter* data;

public:
  AamWriter();
  virtual ~AamWriter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_AAMWRITER_H_ */
