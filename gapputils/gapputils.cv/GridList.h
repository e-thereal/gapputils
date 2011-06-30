#pragma once

#include <gapputils/CombinerInterface.h>
#include <culib/ICudaImage.h>

#include "GridModel.h"

namespace gapputils {

namespace cv {

class GridList : public gapputils::workflow::CombinerInterface {
  InitReflectableClass(GridList)

  Property(ImageNames, std::vector<std::string>)
  Property(Models, boost::shared_ptr<std::vector<boost::shared_ptr<GridModel> > >)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<culib::ICudaImage> > >)
  Property(ImageName, std::string)
  Property(Model, boost::shared_ptr<GridModel>)
  Property(Image, boost::shared_ptr<culib::ICudaImage>)

private:
  static int namesId, modelsId, imagesId;

public:
  GridList();
  virtual ~GridList();

  virtual void clearOutputs();
};

}

}
