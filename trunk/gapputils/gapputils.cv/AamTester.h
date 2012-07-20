#pragma once
#ifndef GAPPUTILSCV_AAMTESTER_H_
#define GAPPUTILSCV_AAMTESTER_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"
#include "GridModel.h"

namespace gapputils {

namespace cv {

class AamTester : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(AamTester)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(SampleImage, boost::shared_ptr<image_t>)
  Property(SampleGrid, boost::shared_ptr<GridModel>)
  Property(FirstMode, float)
  Property(SecondMode, float)

private:
  mutable AamTester* data;

public:
  AamTester(void);
  virtual ~AamTester(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
