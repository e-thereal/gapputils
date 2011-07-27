/*
 * GridImagePair.h
 *
 *  Created on: Jul 27, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_GRIDIMAGEPAIR_H_
#define GAPPUTILSCV_GRIDIMAGEPAIR_H_

#include <gapputils/WorkflowInterface.h>
#include <culib/ICudaImage.h>

#include "GridModel.h"

namespace gapputils {

namespace cv {

class GridImagePair : public gapputils::workflow::WorkflowInterface {
  InitReflectableClass(GridImagePair)

  Property(ImageNames, std::vector<std::string>)
  Property(ImageName, std::string)
  Property(Model, boost::shared_ptr<GridModel>)
  Property(Image, boost::shared_ptr<culib::ICudaImage>)

private:
  static int namesId;

public:
  GridImagePair();
  virtual ~GridImagePair();
};

}

}

#endif /* GAPPUTILSCV_GRIDIMAGEPAIR_H_ */
