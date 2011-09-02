/*
 * AamResample.h
 *
 *  Created on: Sep 2, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMRESAMPLE_H_
#define GAPPUTILSCV_AAMRESAMPLE_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamResample : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamResample)

  Property(InputModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(OutputModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(ColumnCount, int)
  Property(RowCount, int)
  Property(Width, int)
  Property(Height, int)

private:
  mutable AamResample* data;

public:
  AamResample();
  virtual ~AamResample();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_AAMRESAMPLE_H_ */
