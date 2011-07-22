/*
 * AamGenerator.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamGenerator.h"

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

#include <culib/lintrans.h>
#include <culib/CudaImage.h>

#include <algorithm>

#include "ImageWarp.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

DefineEnum(AamGeneratorMode)

BeginPropertyDefinitions(AamGenerator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Input("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Mode, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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

boost::shared_ptr<culib::ICudaImage> createWhiteTexture(dim3 size, dim3 voxelSize = dim3()) {
  boost::shared_ptr<culib::ICudaImage> image(new culib::CudaImage(size, voxelSize));
  float* buffer = image->getOriginalImage();
  const int count = size.x * size.y;
  fill(buffer, buffer + count, 1.0f);
  image->resetWorkingCopy();
  return image;
}

void AamGenerator::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamGenerator();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getParameterVector())
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
  boost::shared_ptr<culib::ICudaImage> image;

  switch(getMode()) {
  case AamGeneratorMode::Image:
    image = model->createTexture(&textureFeatures);
    break;

  case AamGeneratorMode::Segmentation:
    image = createWhiteTexture(dim3(model->getWidth(), model->getHeight()));
    break;
  }

  ImageWarp warp;
  warp.setBaseGrid(model->createMeanShape());
  warp.setWarpedGrid(grid);
  warp.setInputImage(image);
  warp.execute(0);
  warp.writeResults();

  data->setOutputImage(warp.getOutputImage());
}

void AamGenerator::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
