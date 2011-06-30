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

public:
  ActiveAppearanceModel();
  virtual ~ActiveAppearanceModel();

  void featuresFromGrid(std::vector<float>* features, GridModel* grid);
  static void featuresFromImage(std::vector<float>* features, culib::ICudaImage* image);
  static void gridFromFeatures(GridModel* grid, std::vector<float>* features);
  static void imageFromFeatures(culib::ICudaImage* image, std::vector<float>* features);
};

}

}

#endif /* GAPPUTILSCV_ACTIVEAPPEARANCEMODEL_H_ */
