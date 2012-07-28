/*
 * AamCreator.cpp
 *
 *  Created on: Aug 26, 2011
 *      Author: tombr
 */

#include "AamCreator.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <algorithm>

#include <culib/CudaImage.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamCreator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Images, Input("Imgs"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(ShapeParameterCount, Observe(Id), TimeStamp(Id))
  DefineProperty(TextureParameterCount, Observe(Id), TimeStamp(Id))
  DefineProperty(AppearanceParameterCount, Observe(Id), TimeStamp(Id))
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ActiveAppearanceModel, Output("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

AamCreator::AamCreator() : _ShapeParameterCount(1), _TextureParameterCount(1),
    _AppearanceParameterCount(1), _RowCount(2), _ColumnCount(2), data(0)
{
  WfeUpdateTimestamp
  setLabel("AamCreator");

  Changed.connect(capputils::EventHandler<AamCreator>(this, &AamCreator::changedHandler));
}

AamCreator::~AamCreator() {
  if (data)
    delete data;
}

void AamCreator::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {

}

float op_sum (float i, float j) { return i+j; }

struct DivideBy {
  float div;

  DivideBy(float div) : div(div) { }

  float operator()(float& a) const {
    return a / div;
  }
};

void AamCreator::execute(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  if (!data)
    data = new AamCreator();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getImages() && getImages()->size())
    return;

  int spCount, tpCount, apCount, rowCount, columnCount;
  const int width = getImages()->at(0)->getSize()[0], height = getImages()->at(0)->getSize()[1];
  const int pixelCount = width * height;

  // create mean image
  boost::shared_ptr<image_t> meanImage(new image_t(width, height, 1));
  float* meanBuffer = meanImage->getData();

  unsigned imageCount = getImages()->size();
  for (unsigned i = 0; i < imageCount; ++i) {
    image_t* image = getImages()->at(i).get();
    std::transform(meanBuffer, meanBuffer + pixelCount, image->getData(), meanBuffer, op_sum);
  }
  std::transform(meanBuffer, meanBuffer + pixelCount, meanBuffer, DivideBy(imageCount));

  boost::shared_ptr<ActiveAppearanceModel> model(new ActiveAppearanceModel());
  model->setShapeParameterCount(spCount = getShapeParameterCount());
  model->setTextureParameterCount(tpCount = getTextureParameterCount());
  model->setAppearanceParameterCount(apCount = getAppearanceParameterCount());
  model->setRowCount(rowCount = getRowCount());
  model->setColumnCount(columnCount = getColumnCount());
  model->setWidth(width);
  model->setHeight(height);

  boost::shared_ptr<std::vector<float> > meanShape(new std::vector<float>());
  for (int iRow = 0; iRow < rowCount; ++iRow) {
    for (int iCol = 0; iCol < columnCount; ++iCol) {
      meanShape->push_back((float)iCol * width / (columnCount - 1));
      meanShape->push_back((float)iRow * height / (rowCount - 1));
    }
  }
  model->setMeanShape(meanShape);
  model->setMeanTexture(meanImage);

  boost::shared_ptr<std::vector<float> > shapeMatrix(new std::vector<float>(rowCount * columnCount * 2 * spCount));
  std::fill(shapeMatrix->begin(), shapeMatrix->end(), 0.0f);
  model->setShapeMatrix(shapeMatrix);

  boost::shared_ptr<std::vector<float> > textureMatrix(new std::vector<float>(width * height * tpCount));
  std::fill(textureMatrix->begin(), textureMatrix->end(), 0.0f);
  model->setTextureMatrix(textureMatrix);

  boost::shared_ptr<std::vector<float> > appearanceMatrix(new std::vector<float>((spCount + tpCount) * apCount));
  for (int i = 0, k = 0; i < apCount; ++i)
    for (int j = 0; j < spCount + tpCount; ++j, ++k)
      appearanceMatrix->at(k) = (float)(i == j);
  model->setAppearanceMatrix(appearanceMatrix);

  boost::shared_ptr<std::vector<float> > singularShapeParameters(new std::vector<float>(spCount));
  boost::shared_ptr<std::vector<float> > singularTextureParameters(new std::vector<float>(tpCount));
  boost::shared_ptr<std::vector<float> > singularAppearanceParameters(new std::vector<float>(apCount));
  std::fill(singularShapeParameters->begin(), singularShapeParameters->end(), 1.0f);
  std::fill(singularTextureParameters->begin(), singularTextureParameters->end(), 1.0f);
  std::fill(singularAppearanceParameters->begin(), singularAppearanceParameters->end(), 1.0f);
  model->setSingularShapeParameters(singularShapeParameters);
  model->setSingularTextureParameters(singularTextureParameters);
  model->setSingularAppearanceParameters(singularAppearanceParameters);

  data->setActiveAppearanceModel(model);
}

void AamCreator::writeResults() {
  if (!data)
    return;
  setActiveAppearanceModel(data->getActiveAppearanceModel());
}

}

}
