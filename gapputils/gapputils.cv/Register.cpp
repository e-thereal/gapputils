/*
 * Register.cpp
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#include "Register.h"

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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <culib/transform.h>

#include <regutil/CudaImage.h>
#include <regutil/CudaImageTransformation.h>
#include <regutil/SimilarityShannon.h>
#include <regutil/SimilaritySSD.h>
#include <regutil/SimilarityNCC.h>
#include <regutil/TransformationGenerator.h>

#include <culib/util.h>
#include <optlib/ConjugateGradientsOptimizer.h>
#include <optlib/SteepestDescentOptimizer.h>
#include <optlib/SimplifiedPowellOptimizer.h>
#include <optlib/MultistepOptimizer.h>
#include <optlib/DownhillSimplexOptimizer.h>
#include <reglib/RegistrationProblem.h>
#include <optlib/OptimizerException.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Register)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(BaseImage, Input("Base"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage, Input("In"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Similarity, Enumerator<SimilarityMeasure>(), Observe(PROPERTY_ID))
  DefineProperty(Optimizer, Enumerator<OptimizerType>(), Observe(PROPERTY_ID))
  DefineProperty(InPlane, Observe(PROPERTY_ID))
  DefineProperty(Matrix, Output("M"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Register::Register() : _InPlane(true), data(0) {
  WfeUpdateTimestamp
  setLabel("Register");

  Changed.connect(capputils::EventHandler<Register>(this, &Register::changedHandler));
}

Register::~Register() {
  if (data)
    delete data;
}

void Register::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Register::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Register();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage() || !getBaseImage())
    return;

  optlib::IDirectionSetOptimizer::DomainType result;

  if (getInPlane()) {
    result.resize(3);
    for (unsigned i = 0; i < result.size(); ++i) {
      if (i < 3)
        result[i] = 0.0;
      else
        result[i] = 1.0;
    }
  } else {
    result.resize(6);
    for (unsigned i = 0; i < result.size(); ++i) {
      if (i < 6)
        result[i] = 0.0;
      else
        result[i] = 1.0;
    }
  }


  regutil::CudaImage baseImage(getBaseImage()->getSize(), getBaseImage()->getVoxelSize(), getBaseImage()->getWorkingCopy());
  regutil::CudaImage inputImage(getInputImage()->getSize(), getInputImage()->getVoxelSize(), getInputImage()->getWorkingCopy());

  optlib::ObservableDirectionSetOptimizer* optimizer;
  switch(getOptimizer()) {
  case OptimizerType::Powell:
    optimizer = new optlib::MultistepOptimizer<optlib::SimplifiedPowellOptimizer>();
    break;

  /*case OptimizerType::Simplex:
    optimizer = new optlib::MultistepOptimizer<optlib::DownhillSimplexOptimizer>();
    break;
    */
  /*case ConjugateGradients:
    optimizer = new MultistepOptimizer<ConjugateGradientsOptimizer>();
    break;

  case FastConjugateGradients:
    optimizer = new MultistepOptimizer<ConjugateGradientsOptimizer>();
    dynamic_cast<ConjugateGradientsOptimizer*>(optimizer)->setGradientMethod(
        ConjugateGradientsOptimizer::ForwardDifferences);
    break;

  case SimplifiedPowell:
    optimizer = new MultistepOptimizer<SimplifiedPowellOptimizer>();
    break;

  case SteepestDescent:
    optimizer = new MultistepOptimizer<SteepestDescentOptimizer>();
    break;*/

  }

  optimizer->setMaxIterationCount(20);
  //for (unsigned i = 0; i < observers.size(); ++i)
  //  optimizer->addObserver(*observers[i]);

  // Setup multiple scale parameters
  optlib::IMultistepable* multistep = dynamic_cast<optlib::IMultistepable*>(optimizer);
  multistep->addParameter(optlib::XSkip, 4);           // 4.0
  multistep->addParameter(optlib::YSkip, 4);            // 4.0
  multistep->addParameter(optlib::ZSkip, 1);            // 2.0
  multistep->addParameter(optlib::BlurringSigma, 2.0);  // 2.0
  multistep->addParameter(optlib::HistogramSigma, 1.0); // 1.0
  multistep->addParameter(optlib::LineTolerance, 1e-2); // 1e-2
  multistep->addParameter(optlib::Tolerance, 2e-3);     // 1e-2
  multistep->addParameter(optlib::SimilarityMode, culib::SimilarityConfig::JointEntropy | culib::SimilarityConfig::EntropyB | culib::SimilarityConfig::EntropyA);

  multistep->addParameter(optlib::SimilarityMethod, culib::SimilarityConfig::NewMethod);

  multistep->newParameterSet();
  multistep->addParameter(optlib::LineTolerance, 1e-3);
  multistep->addParameter(optlib::Tolerance, 2e-3);
  multistep->addParameter(optlib::XSkip, 2);
  multistep->addParameter(optlib::YSkip, 2);
  multistep->addParameter(optlib::ZSkip, 1);
  multistep->addParameter(optlib::HistogramSigma, 0.0);
  multistep->addParameter(optlib::FreeCaches);      // Force a cache clean up
  multistep->addParameter(optlib::ResetWorkingCopy);
  multistep->addParameter(optlib::BlurringSigma, 1.0);
  multistep->addParameter(optlib::SimilarityMode, culib::SimilarityConfig::JointEntropy | culib::SimilarityConfig::EntropyB | culib::SimilarityConfig::EntropyA);
  
  regutil::TransformationGenerator<regutil::CudaImageTransformation> transGen(baseImage.getSize(), baseImage.getVoxelSize());
  if (getInPlane()) {
    transGen.setZRotationIndex(0);
    transGen.setXTranslationIndex(1);
    transGen.setYTranslationIndex(2);
    transGen.setXScalingIndex(3);
    transGen.setYScalingIndex(4);
  } else {
    transGen.setXRotationIndex(0);
    transGen.setYRotationIndex(1);
    transGen.setZRotationIndex(2);
    transGen.setXTranslationIndex(3);
    transGen.setYTranslationIndex(4);
    transGen.setZTranslationIndex(5);
    transGen.setXScalingIndex(6);
    transGen.setYScalingIndex(7);
    transGen.setZScalingIndex(8);
  }

  //SimilarityShannon simShan(dim3(128, 128), make_float2(16.0, 16.0));
  // TODO: It is assumed that pixel values are in [0,1]
  reglib::ISimilarityMeasure<int3>* similarity = 0;
  switch (getSimilarity()) {
  case SimilarityMeasure::SSD:
    similarity = new regutil::SimilaritySSD();
    break;

  case SimilarityMeasure::NCC:
    similarity = new regutil::SimilarityNCC();
    break;

  case SimilarityMeasure::MI:
    similarity = new regutil::SimilarityShannon(dim3(128, 128), make_float2(1./128.0, 1./128.0));
    break;
  }

  /*if (model.getCenterOfGravity()) {
    float3 cBase = baseImage->getCenterOfGravity();
    float3 cInput = inputImage->getCenterOfGravity();

    std::cout << "Base: " << cBase.x << ", " << cBase.y << ", " << cBase.z << std::endl;
    std::cout << "Input: " << cInput.x << ", " << cInput.y << ", " << cInput.z << std::endl;

    if (model.getInPlane()) {
      result[1] = -cBase.x + cInput.x;
      result[2] = -cBase.y + cInput.y;
    } else {
      result[3] = -cBase.x + cInput.x;
      result[4] = -cBase.y + cInput.y;
      result[5] = -cBase.z + cInput.z;
    }
  }*/

  //if ((resultImage = dynamic_cast<CudaImage*>(transGen.eval(result)->eval(*inputCMif))))
  //  saveAsMif(resultImage, "initial.MIF");

  // Setup the registration problem
  reglib::RegistrationProblem<int3> regprob(&baseImage, &inputImage, &transGen, similarity);
  try {
    regprob.solve(result, *optimizer);
  } catch (optlib::OptimizerException ex) {
    std::cout << "[Warning] Catched exception during registration:" << std::cout;
    std::cout << ex << std::endl;
  }

  //transGen.setParameter(XSkip, 1);
  //transGen.setParameter(YSkip, 1);
  //transGen.setParameter(ZSkip, 1);
  //if ((resultImage = dynamic_cast<CudaImage*>(transGen.eval(result)->eval(*inputCMif))));
  //  saveAsMif(resultImage, "reg.MIF");

  delete optimizer;
  delete similarity;

  regutil::CudaImageTransformation* transform = dynamic_cast<regutil::CudaImageTransformation*>(transGen.eval(result));
  if (transform) {
    data->setMatrix(boost::shared_ptr<fmatrix4>(new fmatrix4(transform->getMatrix(inputImage))));
  }
}

void Register::writeResults() {
  if (!data)
    return;

  setMatrix(data->getMatrix());
}

}

}
