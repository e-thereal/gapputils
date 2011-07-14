/*
 * AamReader.h
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMREADER_H_
#define GAPPUTILSCV_AAMREADER_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamReader : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamReader)

  Property(Filename, std::string)
  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)

private:
  mutable AamReader* data;

public:
  AamReader();
  virtual ~AamReader();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_AAMREADER_H_ */
