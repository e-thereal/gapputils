/*
 * AamGenerator.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamGenerator.h"

#include <capputils/EnumeratorAttribute.h>
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

#include <capputils/HideAttribute.h>

#include <culib/lintrans.h>
#include <culib/CudaImage.h>

#include <algorithm>

#include "ImageWarp.h"

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamGenerator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(ParameterVector, Input("PV"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(BackgroundImage, Input("BG"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(TextureImage, Input("Tex"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(OutputGrid, Output("Grid"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Mode, Enumerator<AamGeneratorMode>(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

AamGenerator::AamGenerator() : _Mode(AamGeneratorMode::Image), data(0) {
  WfeUpdateTimestamp
  setLabel("AamGenerator");

  Changed.connect(capputils::EventHandler<AamGenerator>(this, &AamGenerator::changedHandler));
}

AamGenerator::~AamGenerator() {
  if (data)
    delete data;
}

void AamGenerator::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

boost::shared_ptr<image_t> createWhiteTexture(unsigned width, unsigned height,
    unsigned pixelWidth = 1000, unsigned pixelHeight = 1000)
{
  boost::shared_ptr<image_t> image(new image_t(width, height, 1, pixelWidth, pixelHeight));
  float* buffer = image->getData();
  const int count = width * height;
  fill(buffer, buffer + count, 1.0f);
  return image;
}

void AamGenerator::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamGenerator();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getParameterVector())
      return;

  if ((getMode() == AamGeneratorMode::TextureWarp) && !getTextureImage())
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();

  const int apCount = model->getAppearanceParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  if ((int)getParameterVector()->size() != apCount)
    return;

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float>* parameters = getParameterVector().get();
  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);

  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  culib::lintrans(&modelFeatures[0], &(*model->getAppearanceMatrix())[0], &(*parameters)[0], apCount, 1, spCount + tpCount, false);

  copy(modelFeatures.begin(), modelFeatures.begin() + spCount, shapeParameters.begin());
  copy(modelFeatures.begin() + spCount, modelFeatures.end(), textureParameters.begin());

  culib::lintrans(&shapeFeatures[0], &(*model->getShapeMatrix())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanShape();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  boost::shared_ptr<GridModel> grid = model->createShape(&shapeFeatures);

  culib::lintrans(&textureFeatures[0], &(*model->getTextureMatrix())[0], &textureParameters[0], tpCount, 1, pixelCount, false);
  boost::shared_ptr<vector<float> > meanImage = model->getMeanTexture();
  for (int i = 0; i < pixelCount; ++i)
    textureFeatures[i] = textureFeatures[i] + meanImage->at(i);
  boost::shared_ptr<image_t> image;

  switch(getMode()) {
  case AamGeneratorMode::Image:
    image = model->createTexture(&textureFeatures);
    break;

  case AamGeneratorMode::Segmentation:
    image = createWhiteTexture(model->getWidth(), model->getHeight());
    break;

  case AamGeneratorMode::TextureWarp:
    image = getTextureImage();
    break;
  }

  ImageWarp warp;
  warp.setBaseGrid(model->createMeanShape());
  warp.setWarpedGrid(grid);
  warp.setBackgroundImage(getBackgroundImage());
  warp.setInputImage(image);
  warp.execute(0);
  warp.writeResults();

  data->setOutputImage(warp.getOutputImage());
  data->setOutputGrid(grid);
  grid->freeCaches();
}

void AamGenerator::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
  setOutputGrid(data->getOutputGrid());
}

}

}
