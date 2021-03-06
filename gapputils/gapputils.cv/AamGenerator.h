/*
 * AamGenerator.h
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMGENERATOR_H_
#define GAPPUTILSCV_AAMGENERATOR_H_

#include <gapputils/WorkflowElement.h>

#include <capputils/Enumerators.h>

#include "ActiveAppearanceModel.h"
#include "GridModel.h"

namespace gapputils {

namespace cv {

CapputilsEnumerator(AamGeneratorMode, Image, Segmentation, TextureWarp);

class AamGenerator : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamGenerator)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(ParameterVector, boost::shared_ptr<std::vector<float> >)
  Property(BackgroundImage, boost::shared_ptr<image_t>)
  Property(TextureImage, boost::shared_ptr<image_t>)
  Property(OutputImage, boost::shared_ptr<image_t>)
  Property(OutputGrid, boost::shared_ptr<GridModel>)
  Property(Mode, AamGeneratorMode)

private:
  mutable AamGenerator* data;

public:
  AamGenerator();
  virtual ~AamGenerator();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}


#endif /* GAPPUTILSCV_AAMGENERATOR_H_ */
