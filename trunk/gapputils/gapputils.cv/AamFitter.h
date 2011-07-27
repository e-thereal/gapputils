/*
 * AamFitter.h
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMFITTER_H_
#define GAPPUTILSCV_AAMFITTER_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamFitter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamFitter)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(ParameterVector, boost::shared_ptr<std::vector<float> >)
  Property(Similarity, double)

private:
  mutable AamFitter* data;

public:
  AamFitter();
  virtual ~AamFitter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_AAMFITTER_H_ */
