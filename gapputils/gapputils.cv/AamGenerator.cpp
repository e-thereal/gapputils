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

#include <algorithm>

#include "ImageWarp.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamGenerator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Input("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamGenerator::AamGenerator() : data(0) {
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

void AamGenerator::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamGenerator();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getParameterVector())
      return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();

  const int mpCount = model->getModelParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  if (getParameterVector()->size() != mpCount)
    return;

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float>* parameters = getParameterVector().get();
  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);

  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  culib::lintrans(&modelFeatures[0], &(*model->getPrincipalParameters())[0], &(*parameters)[0], mpCount, 1, spCount + tpCount, false);

  copy(modelFeatures.begin(), modelFeatures.begin() + spCount, shapeParameters.begin());
  copy(modelFeatures.begin() + spCount, modelFeatures.end(), textureParameters.begin());

  culib::lintrans(&shapeFeatures[0], &(*model->getPrincipalGrids())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanGrid();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  boost::shared_ptr<GridModel> grid = model->createGrid(&shapeFeatures);

  culib::lintrans(&textureFeatures[0], &(*model->getPrincipalImages())[0], &textureParameters[0], tpCount, 1, pixelCount, false);
  boost::shared_ptr<vector<float> > meanImage = model->getMeanImage();
  for (int i = 0; i < pixelCount; ++i)
    textureFeatures[i] = textureFeatures[i] + meanImage->at(i);
  boost::shared_ptr<culib::ICudaImage> image = model->createImage(&textureFeatures);

  ImageWarp warp;
  warp.setBaseGrid(model->createMeanGrid());
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
