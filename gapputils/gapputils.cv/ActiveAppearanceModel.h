/*
 * ActiveAppearanceModel.h
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_ACTIVEAPPEARANCEMODEL_H_
#define GAPPUTILSCV_ACTIVEAPPEARANCEMODEL_H_

#include <capputils/ReflectableClass.h>

#include <culib/ICudaImage.h>

#include "GridModel.h"

namespace gapputils {

namespace cv {

class ActiveAppearanceModel : public capputils::reflection::ReflectableClass {
  InitReflectableClass(ActiveAppearanceModel)

  Property(MeanGrid, boost::shared_ptr<std::vector<float> >)
  Property(MeanImage, boost::shared_ptr<std::vector<float> >)
  Property(PrincipalGrids, boost::shared_ptr<std::vector<float> >)
  Property(PrincipalImages, boost::shared_ptr<std::vector<float> >)
  Property(PrincipalParameters, boost::shared_ptr<std::vector<float> >)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Width, int)
  Property(Height, int)
  Property(ShapeParameterCount, int)
  Property(TextureParameterCount, int)
  Property(ModelParameterCount, int)

public:
  ActiveAppearanceModel();
  virtual ~ActiveAppearanceModel();

  boost::shared_ptr<GridModel> createMeanGrid();
  boost::shared_ptr<culib::ICudaImage> createMeanImage();
  boost::shared_ptr<GridModel> createGrid(std::vector<float>* features);
  boost::shared_ptr<culib::ICudaImage> createImage(std::vector<float>* features);

  void setMeanGrid(boost::shared_ptr<GridModel> grid);
  void setMeanImage(boost::shared_ptr<culib::ICudaImage> image);

  static boost::shared_ptr<std::vector<float> > toFeatures(boost::shared_ptr<GridModel> grid);
  static boost::shared_ptr<std::vector<float> > toFeatures(boost::shared_ptr<culib::ICudaImage> image);
};

}

}

#endif /* GAPPUTILSCV_ACTIVEAPPEARANCEMODEL_H_ */
