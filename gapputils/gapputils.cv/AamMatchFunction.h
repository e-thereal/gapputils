#pragma once
#ifndef GAPPUTILSCV_AAMMATCHFUNCTION_H_
#define GAPPUTILSCV_AAMMATCHFUNCTION_H_

#include <optlib/IMultiDimensionOptimizer.h>

#include <boost/shared_ptr.hpp>
#include <culib/similarity.h>

#include "../gapputils.cv.cuda/AamMatchFunction.h"

#include <thrust/device_vector.h>

#include "ActiveAppearanceModel.h"

#include <capputils/Enumerators.h>
#include "SimilarityMeasure.h"

namespace gapputils {

namespace cv {

class AamMatchFunction : public virtual optlib::IFunction<optlib::IMultiDimensionOptimizer::DomainType>
{
private:
  boost::shared_ptr<image_t> image;
  boost::shared_ptr<ActiveAppearanceModel> model;
  culib::SimilarityConfig config;
  bool inReferenceFrame;
  SimilarityMeasure measure;
  bool useAm;
  int pointCount, pixelCount, spCount, tpCount, apCount;
  cuda::AamMatchStatus status;

  boost::shared_ptr<image_t> warpedImage;
  thrust::device_vector<float> d_shapeMatrix;
  thrust::device_vector<float> d_textureMatrix;
  thrust::device_vector<float> d_appearanceMatrix;
  thrust::device_vector<float> d_meanShape;
  thrust::device_vector<float> d_meanTexture;

public:
  AamMatchFunction(boost::shared_ptr<image_t> image,
      boost::shared_ptr<ActiveAppearanceModel> model, bool inReferenceFrame,
      SimilarityMeasure measure, bool useAm);
  virtual ~AamMatchFunction(void);

  virtual double eval(const DomainType& parameter);
};

}

}

#endif /* GAPPUTILSCV_AAMMATCHFUNCTION_H_ */
