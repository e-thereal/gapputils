/*
 * AamBuilder.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMBUILDER_H_
#define GAPPUTILSCV_AAMBUILDER_H_

#include <gapputils/WorkflowElement.h>

#include <gapputils/Image.h>

#include "GridModel.h"
#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamBuilder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamBuilder)

  Property(Grids, boost::shared_ptr<std::vector<boost::shared_ptr<GridModel> > >)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)

  Property(ShapeParameterCount, int)
  Property(TextureParameterCount, int)
  Property(AppearanceParameterCount, int)

//  Property(MeanGrid, boost::shared_ptr<GridModel>)
  Property(MeanImage, boost::shared_ptr<image_t>)

private:
  mutable AamBuilder* data;

public:
  AamBuilder();
  virtual ~AamBuilder();

  // calculates the mean grid and the principal grids
  // calculates the mean image and all principal images
  // calculates the final AAM parameters by applying PCA to shape and texture parameters
  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILSCV_AAMBUILDER_H_ */
