/*
 * AamBuilder2.cpp
 *
 *  Created on: Jul 20, 2011
 *      Author: tombr
 */

#include "AamBuilder2.h"

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

#include <optlib/DownhillSimplexOptimizer.h>

#include "AamEcdnll.h"
#include "AamFitter.h"
#include "GridModel.h"
#include "AamBuilder.h"
#include "AamWriter.h"
#include "AamUtils.h"

#include <cassert>

#include <algorithm>
#include <iostream>
#include <sstream>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

#define TRACE std::cout << __LINE__ << std::endl;

namespace gapputils {

namespace cv {

DefineEnum(AamBuilderMode)

BeginPropertyDefinitions(AamBuilder2)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(TrainingSet, Input("D"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InitialModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Mode, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ModelQuality, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IterationCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

  DefineProperty(ActiveAppearanceModel, Output("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamBuilder2::AamBuilder2() : _ModelQuality(0), _IterationCount(5), data(0) {
  WfeUpdateTimestamp
  setLabel("AamBuilder2");

  Changed.connect(capputils::EventHandler<AamBuilder2>(this, &AamBuilder2::changedHandler));
}

AamBuilder2::~AamBuilder2() {
  if (data)
    delete data;
}

void AamBuilder2::changedHandler(capputils::ObservableClass* /*sender*/, int /*eventId*/) {

}

void AamBuilder2::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamBuilder2();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int maxIter = 20;
  float variance = 1.0f, lambda = 1.0f;
  vector<float> sigma;

  const int shapeFeatureCount = getInitialModel()->getColumnCount() * getInitialModel()->getRowCount() * 2;
  const int shapeMatrixSize = shapeFeatureCount * getInitialModel()->getShapeParameterCount();

  boost::shared_ptr<ActiveAppearanceModel> result, initialModel;
  boost::shared_ptr<vector<boost::shared_ptr<culib::ICudaImage> > > trainingSet = getTrainingSet();
  initialModel = getInitialModel();

  if (getMode() == AamBuilderMode::Build) {
    for (int i = 0; i < maxIter; ++i) {
      // Minimize ecdnll -> get new shape model
      // use new shape model and old texture model to find best grids (Fitter gives shape parameters)
      // Use grids and training set to update the shape and texture model (Using default model builder)

      assert(initialModel);
      assert(initialModel->getShapeMatrix());
      assert((int)initialModel->getShapeMatrix()->size() == shapeMatrixSize);
      assert((int)(initialModel->getShapeMatrix()->end() - initialModel->getShapeMatrix()->begin()) == shapeMatrixSize);

      AamEcdnll ecdnll(trainingSet, initialModel, variance, lambda, sigma);
      vector<double> parameter(shapeFeatureCount + shapeMatrixSize);
      std::copy(initialModel->getMeanShape()->begin(), initialModel->getMeanShape()->end(),
          parameter.begin());
      std::copy(initialModel->getShapeMatrix()->begin(),
          initialModel->getShapeMatrix()->end(),
          parameter.begin() + shapeFeatureCount);

      optlib::DownhillSimplexOptimizer optimizer;
      optimizer.minimize(parameter, ecdnll);

      // Create new model using updated shape model and old texture model
      result = ecdnll.updateModel(parameter);
      double fitValue = ecdnll.eval(parameter);

      // Fit every image with the current model and build grids according to the best fit
      AamFitter fitter;
      fitter.setActiveAppearanceModel(result);
      fitter.setInReferenceFrame(true);
      fitter.setMeasure(SimilarityMeasure::SSD);
      fitter.setUseAppearanceMatrix(false);
      vector<float> shapeFeatures(shapeFeatureCount);

      boost::shared_ptr<vector<boost::shared_ptr<GridModel> > > grids(new vector<boost::shared_ptr<GridModel> >());
      for (unsigned iImage = 0; iImage < trainingSet->size(); ++iImage) {
        fitter.setInputImage(trainingSet->at(iImage));
        fitter.execute(0);
        fitter.writeResults();

        AamUtils::getShapeFeatures(&shapeFeatures, result.get(), fitter.getShapeParameters().get());
        grids->push_back(result->createShape(&shapeFeatures));
      }

      // Rebuild the active appearance model using the training set and the fitted grids
      AamBuilder builder;
      builder.setAppearanceParameterCount(initialModel->getAppearanceParameterCount());
      builder.setTextureParameterCount(initialModel->getTextureParameterCount());
      builder.setShapeParameterCount(initialModel->getShapeParameterCount());
      builder.setImages(trainingSet);
      builder.setGrids(grids);
      builder.execute(0);
      builder.writeResults();
      initialModel = builder.getActiveAppearanceModel();

      // Save model for debugging purpose
      static int iterationCounter = 0;
      std::cout << "Saving new model..." << std::flush;
      AamWriter writer;
      std::stringstream filename;
      filename << "AAMs/new_" << iterationCounter++ << " (" << fitValue << ").amm";
      writer.setFilename(filename.str());
      writer.setActiveAppearanceModel(initialModel);
      writer.execute(0);
      writer.writeResults();
      std::cout << " done!" << std::endl;

      if (monitor) monitor->reportProgress(100 * (i + 1) / maxIter);
    }
  }

  // Calculate quality of fit
  assert(initialModel);
  assert(initialModel->getShapeMatrix());
  assert((int)initialModel->getShapeMatrix()->size() == shapeMatrixSize);
  assert((int)(initialModel->getShapeMatrix()->end() - initialModel->getShapeMatrix()->begin()) == shapeMatrixSize);

  AamEcdnll ecdnll(trainingSet, initialModel, variance, lambda, sigma);
  vector<double> parameter(shapeFeatureCount + shapeMatrixSize);
  std::copy(initialModel->getMeanShape()->begin(), initialModel->getMeanShape()->end(),
      parameter.begin());
  std::copy(initialModel->getShapeMatrix()->begin(),
      initialModel->getShapeMatrix()->end(),
      parameter.begin() + shapeFeatureCount);

  data->setModelQuality(ecdnll.eval(parameter));
  data->setActiveAppearanceModel(initialModel);
}

void AamBuilder2::writeResults() {
  if (!data)
    return;
  setActiveAppearanceModel(data->getActiveAppearanceModel());
  setModelQuality(data->getModelQuality());
}

}

}
