/*
 * AamFitter.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "AamFitter.h"

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

#include <culib/lintrans.h>

#include <capputils/HideAttribute.h>

#include <optlib/DownhillSimplexOptimizer.h>
#include <optlib/SimplifiedPowellOptimizer.h>
#include <optlib/SteepestDescentOptimizer.h>
#include <optlib/OptimizerException.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

#include "AamMatchFunction.h"
#include "AamGenerator.h"
#include "ImageWarp.h"
#include "AamWriter.h"

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamFitter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(InputImage, Input("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Measure, Enumerator<SimilarityMeasure>(), Observe(Id), TimeStamp(Id))
  DefineProperty(InReferenceFrame, Observe(Id), TimeStamp(Id))
  DefineProperty(UseAppearanceMatrix, Observe(Id), TimeStamp(Id))
  DefineProperty(AppearanceParameters, Output("AP"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(ShapeParameters, Output("SP"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Similarity, Output("Sim"), Observe(Id), TimeStamp(Id))
EndPropertyDefinitions

AamFitter::AamFitter() : _InReferenceFrame(true), _UseAppearanceMatrix(true), data(0) {
  WfeUpdateTimestamp
  setLabel("AamFitter");

  Changed.connect(capputils::EventHandler<AamFitter>(this, &AamFitter::changedHandler));
}

AamFitter::~AamFitter() {
  if (data)
    delete data;
}

void AamFitter::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {

}

void AamFitter::execute(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  if (!data)
    data = new AamFitter();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();
  boost::shared_ptr<image_t> image = getInputImage();

  if (!model || !image)
    return;

  AamMatchFunction objective(getInputImage(), model, getInReferenceFrame(),
      getMeasure(), getUseAppearanceMatrix());

  //optlib::DownhillSimplexOptimizer optimizer;
  //optlib::SimplifiedPowellOptimizer optimizer;
  optlib::SteepestDescentOptimizer optimizer;
  std::vector<double> parameter(getActiveAppearanceModel()->getShapeParameterCount());
  try {
    optimizer.maximize(parameter, objective);
  } catch (optlib::OptimizerException ex) {
    static int exceptionCount = 0;
    cout << "[AamFitter: " << __LINE__ << "] Exception caught: " << ex.what() << endl;
    stringstream filename;
    filename << "exception_" << exceptionCount++ << ".aam";
    AamWriter writer;
    writer.setActiveAppearanceModel(getActiveAppearanceModel());
    writer.setFilename(filename.str());
    writer.execute(0);
    writer.writeResults();
  }

  // Get model parameters for shape parameters
  // - warp image to reference frame using current shape parameters
  // - get texture parameters for best match
  // - calculate possible model parameters
  const int apCount = model->getAppearanceParameterCount();
  const int spCount = model->getShapeParameterCount();
  const int tpCount = model->getTextureParameterCount();

  const int pixelCount = model->getWidth() * model->getHeight();
  const int pointCount = model->getColumnCount() * model->getRowCount();

  boost::shared_ptr<vector<float> > shapeParameters(new vector<float>(spCount));
  vector<float> textureParameters(tpCount);
  boost::shared_ptr<vector<float> > appearanceParameters(new vector<float>(apCount));

  vector<float> modelFeatures(spCount + tpCount);
  vector<float> shapeFeatures(2 * pointCount);
  vector<float> textureFeatures(pixelCount);

  ImageWarp warp;
  warp.setWarpedGrid(model->createMeanShape());
  warp.setInputImage(image);

  copy(parameter.begin(), parameter.end(), shapeParameters->begin());
  culib::lintrans(&shapeFeatures[0], &(*model->getShapeMatrix())[0], &(*shapeParameters)[0], spCount, 1, 2 * pointCount, false);
  boost::shared_ptr<vector<float> > meanGrid = model->getMeanShape();
  for (int i = 0; i < 2 * pointCount; ++i)
    shapeFeatures[i] = shapeFeatures[i] + meanGrid->at(i);
  warp.setBaseGrid(model->createShape(&shapeFeatures));
  warp.execute(0);
  warp.writeResults();

  boost::shared_ptr<vector<float> > imageFeatures = model->toFeatures(warp.getOutputImage().get());
  boost::shared_ptr<vector<float> > meanImageFeatures = model->getMeanTexture();
  for (int i = 0; i < pixelCount; ++i)
    (*imageFeatures)[i] = imageFeatures->at(i) - meanImageFeatures->at(i);
  culib::lintrans(&textureParameters[0], &(*model->getTextureMatrix())[0], &(*imageFeatures)[0], pixelCount, 1, tpCount, true);

  copy(shapeParameters->begin(), shapeParameters->end(), modelFeatures.begin());
  copy(textureParameters.begin(), textureParameters.end(), modelFeatures.begin() + spCount);
  culib::lintrans(&(*appearanceParameters)[0], &(*model->getAppearanceMatrix())[0], &modelFeatures[0], spCount + tpCount, 1, apCount, true);

  data->setAppearanceParameters(appearanceParameters);
  data->setShapeParameters(shapeParameters);
  data->setSimilarity(objective.eval(parameter));
}

void AamFitter::writeResults() {
  if (!data)
    return;

  setAppearanceParameters(data->getAppearanceParameters());
  setShapeParameters(data->getShapeParameters());
  setSimilarity(data->getSimilarity());
}

}

}
