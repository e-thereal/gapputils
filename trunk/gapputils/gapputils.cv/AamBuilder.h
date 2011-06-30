/*
 * AamBuilder.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMBUILDER_H_
#define GAPPUTILSCV_AAMBUILDER_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>

#include "GridModel.h"
#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamBuilder : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamBuilder)

  Property(Grids, boost::shared_ptr<std::vector<boost::shared_ptr<GridModel> > >)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)

//  Property(MeanGrid, boost::shared_ptr<GridModel>)
  Property(MeanImage, boost::shared_ptr<culib::ICudaImage>)

private:
  mutable AamBuilder* data;

public:
  AamBuilder();
  virtual ~AamBuilder();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

private:
  // calculates the mean grid and the principal grids
  void calculateGrids() const;

  // calculates the mean image and all principal images
  void calculateImages() const;
};

}

}

#endif /* GAPPUTILSCV_AAMBUILDER_H_ */
