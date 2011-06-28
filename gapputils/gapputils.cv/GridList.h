#pragma once

#include <gapputils/CombinerInterface.h>
#include <culib/ICudaImage.h>

#include "GridModel.h"

namespace gapputils {

namespace cv {

class GridList : public gapputils::workflow::CombinerInterface {
  InitReflectableClass(GridList)

  Property(ImageNames, std::vector<std::string>)
  Property(Models, std::vector<GridModel*>*)
  Property(ImageName, std::string)
  Property(Model, GridModel*)
  Property(Image, culib::ICudaImage*)

private:
  static int namesId, modelsId;

public:
  GridList();
  virtual ~GridList();

  virtual void clearOutputs();
};

}

}