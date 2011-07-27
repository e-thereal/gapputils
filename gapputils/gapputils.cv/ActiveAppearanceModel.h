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

  Property(MeanShape, boost::shared_ptr<std::vector<float> >)
  Property(MeanTexture, boost::shared_ptr<std::vector<float> >)
  Property(ShapeMatrix, boost::shared_ptr<std::vector<float> >)
  Property(TextureMatrix, boost::shared_ptr<std::vector<float> >)
  Property(AppearanceMatrix, boost::shared_ptr<std::vector<float> >)

  Property(SingularShapeParameters, boost::shared_ptr<std::vector<float> >)
  Property(SingularTextureParameters, boost::shared_ptr<std::vector<float> >)
  Property(SingularAppearanceParameters, boost::shared_ptr<std::vector<float> >)

  Property(RowCount, int)
  Property(ColumnCount, int)
  Property(Width, int)
  Property(Height, int)
  Property(ShapeParameterCount, int)
  Property(TextureParameterCount, int)
  Property(AppearanceParameterCount, int)

public:
  ActiveAppearanceModel();
  virtual ~ActiveAppearanceModel();

  boost::shared_ptr<GridModel> createMeanShape();
  boost::shared_ptr<culib::ICudaImage> createMeanTexture();
  boost::shared_ptr<GridModel> createShape(std::vector<float>* features);
  boost::shared_ptr<culib::ICudaImage> createTexture(std::vector<float>* features);

  void setMeanShape(boost::shared_ptr<GridModel> grid);
  void setMeanTexture(boost::shared_ptr<culib::ICudaImage> image);

  static boost::shared_ptr<std::vector<float> > toFeatures(GridModel* grid);
  static boost::shared_ptr<std::vector<float> > toFeatures(culib::ICudaImage* image);
};

}

}

#endif /* GAPPUTILSCV_ACTIVEAPPEARANCEMODEL_H_ */
