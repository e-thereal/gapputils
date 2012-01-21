/*
 * FgrbmDecoder.h
 *
 *  Created on: Jan 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMDECODER_H_
#define GAPPUTILS_ML_FGRBMDECODER_H_

#include <gapputils/WorkflowElement.h>

#include "FgrbmModel.h"

namespace gapputils {

namespace ml {

class FgrbmDecoder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FgrbmDecoder)

    Property(FgrbmModel, boost::shared_ptr<FgrbmModel>)
    Property(ConditionalVector, boost::shared_ptr<std::vector<double> >)
    Property(HiddenVector, boost::shared_ptr<std::vector<double> >)
    Property(VisibleVector, boost::shared_ptr<std::vector<double> >)
    Property(SampleVisibles, bool)
    Property(IsGaussian, bool)

private:
  mutable FgrbmDecoder* data;

public:
  FgrbmDecoder();
  virtual ~FgrbmDecoder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_ML_FGRBMDECODER_H_ */
