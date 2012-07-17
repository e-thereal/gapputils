/*
 * AamCreator.h
 *
 *  Created on: Aug 26, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMCREATOR_H_
#define GAPPUTILSCV_AAMCREATOR_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamCreator : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamCreator)

  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(ShapeParameterCount, int)
  Property(TextureParameterCount, int)
  Property(AppearanceParameterCount, int)
  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)

private:
  mutable AamCreator* data;

public:
  AamCreator();
  virtual ~AamCreator();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_AAMCREATOR_H_ */
