/*
 * AamBuilder2.h
 *
 *  Created on: Jul 20, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMBUILDER2_H_
#define GAPPUTILSCV_AAMBUILDER2_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Enumerators.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

CapputilsEnumerator(AamBuilderMode, Build, Evaluate);

class AamBuilder2 : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamBuilder2)

  Property(TrainingSet, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(InitialModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(Mode, AamBuilderMode)
  Property(ModelQuality, double)
  Property(IterationCount, int)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)

private:
  mutable AamBuilder2* data;

public:
  AamBuilder2();
  virtual ~AamBuilder2();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_AAMBUILDER2_H_ */
