/*
 * AamFitter.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamFitter.h"

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

#include <culib/lintrans.h>

#include <gapputils/HideAttribute.h>

#include <optlib/DownhillSimplexOptimizer.h>

#include <algorithm>
#include <cassert>
#include <iostream>

#include "AamMatchFunction.h"
#include "AamGenerator.h"
#include "ImageWarp.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamFitter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Output("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SSD, Output(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
EndPropertyDefinitions

AamFitter::AamFitter() : data(0) {
  WfeUpdateTimestamp
  setLabel("AamFitter");

  Changed.connect(capputils::EventHandler<AamFitter>(this, &AamFitter::changedHandler));
}

AamFitter::~AamFitter() {
  if (data)
    delete data;
}

void AamFitter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void AamFitter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamFitter();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();
  boost::shared_ptr<culib::ICudaImage> image = getInputImage();

  if (!model || !image)
    return;

  AamMatchFunction objective(getInputImage(), model);

  optlib::DownhillSimplexOptimizer optimizer;
  std::vector<double> parameter(getActiveAppearanceModel()->getShapeParameterCount());
  optimizer.maximize(parameter, objective);

  // Get model parameters for shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  const int apCount = model->getAppearanceParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  vector<float> shapeParameters(spCount);
  vector<float> textureParameters(tpCount);
  vector<float> modelParameters(apCount);

  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  ImageWarp warp;
  warp.setWarpedGrid(model->createMeanShape());
  warp.setInputImage(image);

  copy(parameter.begin(), parameter.end(), shapeParameters.begin());
  culib::lintrans(&shapeFeatures[0], &(*model->getShapeMatrix())[0], &shapeParameters[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanShape();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  warp.setBaseGrid(model->createShape(&shapeFeatures));
  warp.execute(0);
  warp.writeResults();

  boost::shared_ptr<vector<float> > imageFeatures = model->toFeatures(warp.getOutputImage());
  boost::shared_ptr<vector<float> > meanImageFeatures = model->getMeanTexture();
  for (int i = 0; i < pixelCount; ++i)
    (*imageFeatures)[i] = imageFeatures->at(i) - meanImageFeatures->at(i);
  culib::lintrans(&textureParameters[0], &(*model->getTextureMatrix())[0], &(*imageFeatures)[0], pixelCount, 1, tpCount, true);

  copy(shapeParameters.begin(), shapeParameters.end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&modelParameters[0], &(*model->getAppearanceMatrix())[0], &modelFeatures[0], spCount + tpCount, 1, apCount, true);

  boost::shared_ptr<vector<float> > pv(new vector<float>(apCount));
  copy(modelParameters.begin(), modelParameters.end(), pv->begin());
  data->setParameterVector(pv);
  data->setSSD(-objective.eval(parameter));
}

void AamFitter::writeResults() {
  if (!data)
    return;

  setParameterVector(data->getParameterVector());
  setSSD(data->getSSD());
}

}

}
