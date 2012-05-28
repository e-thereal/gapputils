/*
 * MifregApplication.cpp
 *
 *  Created on: Nov 25, 2008
 *      Author: tombr
 */

#include "MifregEngine.h"

#include "constants.h"

#include "ICudaImage.h"
#include "CudaImageTransformation.h"
#include "SimilarityShannon.h"
#include "TransformationGenerator.h"

#include <culib/util.h>
#include <optlib/ConjugateGradientsOptimizer.h>
#include <optlib/SteepestDescentOptimizer.h>
#include <optlib/SimplifiedPowellOptimizer.h>
#include <optlib/MultistepOptimizer.h>
#include <reglib/RegistrationProblem.h>

#include <iostream>
#include <cassert>

//#include "DataModel.h"

//#include <CMIF.hpp>
//#include <CChannel.hpp>

#include <optlib/OptimizerException.h>

using namespace culib;
using namespace optlib;
using namespace reglib;

using namespace std;

namespace regutil {

MifregEngine::MifregEngine() : baseCMif(0), inputCMif(0),
    algorithm(ConjugateGradients), simMeasure(NewMethod) { }

void MifregEngine::addObserver(ObserverType& observer) {
  observers.push_back(&observer);
}

void MifregEngine::removeObserver(ObserverType& observer) {
  bool found = false;

  for (unsigned i = 0; i < observers.size(); ++i) {
    if (found)
      observers[i-1] = observers[i];
    else if (observers[i] == &observer)
      found = true;
  }

  if (found)
    observers.resize(observers.size()-1);
}

void MifregEngine::setBaseCMif(ICudaImage *cmif) {
  baseCMif = cmif;
}

void MifregEngine::setInputCMif(ICudaImage *cmif) {
  inputCMif = cmif;
}

void MifregEngine::setAlgorithm(OptimizationAlgorithm algorithm) {
  this->algorithm = algorithm;
}

MifregEngine::OptimizationAlgorithm MifregEngine::getAlgorithm() {
  return algorithm;
}

void MifregEngine::setSimilarityMeasure(SimilarityMeasure simMeasure) {
  this->simMeasure = simMeasure;
}

MifregEngine::SimilarityMeasure MifregEngine::getSimilarityMeasure() {
  return simMeasure;
}

void MifregEngine::initializeResult(IDirectionSetOptimizer::DomainType& result) {
  //DataModel& model = DataModel::getInstance();

  /*if (model.getInPlane()) {
    for (unsigned i = 0; i < result.size(); ++i) {
      if (i < 3)
        result[i] = 0.0;
      else
        result[i] = 1.0;
    }
  } else {*/
    for (unsigned i = 0; i < result.size(); ++i) {
      if (i < 6)
        result[i] = 0.0;
      else
        result[i] = 1.0;
    }
  //}
}

void MifregEngine::performRegistration(IDirectionSetOptimizer::DomainType& result)
{
  CudaImage* baseImage = dynamic_cast<CudaImage*>(baseCMif);
  CudaImage* inputImage = dynamic_cast<CudaImage*>(inputCMif);
  CudaImage* resultImage = 0;

  //DataModel& model = DataModel::getInstance();

  assert(baseImage);
  assert(inputImage);

  ObservableDirectionSetOptimizer* optimizer;
  switch(algorithm) {
  case ConjugateGradients:
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
    break;
  }

  optimizer->setMaxIterationCount(20);
  for (unsigned i = 0; i < observers.size(); ++i)
    optimizer->addObserver(*observers[i]);

  // Setup multiple scale parameters
  IMultistepable* multistep = dynamic_cast<IMultistepable*>(optimizer);
  multistep->addParameter(XSkip, 4);           // 4.0
  multistep->addParameter(YSkip, 4);            // 4.0
  multistep->addParameter(ZSkip, 3);            // 2.0
  multistep->addParameter(BlurringSigma, 2.0);  // 2.0
  multistep->addParameter(HistogramSigma, 1.0); // 1.0
  multistep->addParameter(LineTolerance, 1e-2); // 1e-2
  multistep->addParameter(Tolerance, 2e-3);     // 1e-2
  multistep->addParameter(SimilarityMode, SimilarityConfig::JointEntropy | SimilarityConfig::EntropyB | SimilarityConfig::EntropyA);

  /*switch (simMeasure) {
    case OldMethod:   multistep->addParameter(SimilarityMethod, SimilarityConfig::OldMethod); break;
    case NewMethod:   multistep->addParameter(SimilarityMethod, SimilarityConfig::NewMethod); break;
    case ShamsMethod1: multistep->addParameter(SimilarityMethod, SimilarityConfig::ShamsMethod1); break;
    case ShamsMethod2: multistep->addParameter(SimilarityMethod, SimilarityConfig::ShamsMethod2); break;
    
    default: multistep->addParameter(SimilarityMethod, SimilarityConfig::NewMethod); break;
  }*/
  multistep->addParameter(SimilarityMethod, SimilarityConfig::OldMethod);

  multistep->newParameterSet();
  multistep->addParameter(LineTolerance, 1e-3);
  multistep->addParameter(Tolerance, 2e-3);
  multistep->addParameter(XSkip, 2);
  multistep->addParameter(YSkip, 2);
  multistep->addParameter(ZSkip, 1);
  multistep->addParameter(HistogramSigma, 0.0);
  multistep->addParameter(FreeCaches);      // Force a cache clean up
  multistep->addParameter(ResetWorkingCopy);
  multistep->addParameter(BlurringSigma, 1.0);
  multistep->addParameter(SimilarityMode, SimilarityConfig::JointEntropy | SimilarityConfig::EntropyB | SimilarityConfig::EntropyA);
  
  TransformationGenerator<CudaImageTransformation> transGen(baseCMif->getSize(), baseCMif->getVoxelSize());
  /*if (model.getInPlane()) {
    transGen.setZRotationIndex(0);
    transGen.setXTranslationIndex(1);
    transGen.setYTranslationIndex(2);
    transGen.setXScalingIndex(3);
    transGen.setYScalingIndex(4);
  } else {*/
    transGen.setXRotationIndex(0);
    transGen.setYRotationIndex(1);
    transGen.setZRotationIndex(2);
    transGen.setXTranslationIndex(3);
    transGen.setYTranslationIndex(4);
    transGen.setZTranslationIndex(5);
    transGen.setXScalingIndex(6);
    transGen.setYScalingIndex(7);
    transGen.setZScalingIndex(8);
  //}

  //SimilarityShannon simShan(dim3(128, 128), make_float2(16.0, 16.0));
  // TODO: It is assumed that pixel values are in [0,1]
  SimilarityShannon simShan(dim3(128, 128), make_float2(1./128.0, 1./128.0));

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
  RegistrationProblem<int3> regprob(baseCMif, inputCMif, &transGen, &simShan);
  try {
    //if (!model.getInitializeOnly())
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
  baseCMif->resetWorkingCopy();
  inputCMif->resetWorkingCopy();
}

ostream& operator<<(ostream& os, const MifregEngine::OptimizationAlgorithm& algorithm) {
  switch (algorithm) {
  case MifregEngine::ConjugateGradients:
    os << "Conjugate Gradients";
    break;

  case MifregEngine::FastConjugateGradients:
    os << "Fast Conjugate Gradients";
    break;

  case MifregEngine::SimplifiedPowell:
    os << "Powell";
    break;

  case MifregEngine::SteepestDescent:
    os << "Steepest Descent";
    break;

  default:
    os << "Unknown Algorithm";
  }
  return os;
}

ostream& operator<<(ostream& os, const MifregEngine::SimilarityMeasure& simMeasure) {
  switch (simMeasure) {
  case MifregEngine::OldMethod:
    os << "Old Method";
    break;
 
  case MifregEngine::NewMethod:
    os << "New Method";
    break;
    
  case MifregEngine::ShamsMethod1:
    os << "Shams' Method I";
    break;
    
  case MifregEngine::ShamsMethod2:
    os << "Shams' Method II";
    break;

  default:
    os << "Unknown Similarity Measure";
  }
  return os;
}

}
