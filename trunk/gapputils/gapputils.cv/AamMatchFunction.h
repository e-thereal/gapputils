#pragma once
#ifndef GAPPUTILSCV_AAMMATCHFUNCTION_H_
#define GAPPUTILSCV_AAMMATCHFUNCTION_H_

#include <optlib/IMultiDimensionOptimizer.h>

#include <boost/shared_ptr.hpp>
#include <culib/ICudaImage.h>
#include <culib/similarity.h>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

class AamMatchFunction : public virtual optlib::IFunction<optlib::IMultiDimensionOptimizer::DomainType>
{
private:
  boost::shared_ptr<culib::ICudaImage> image;
  boost::shared_ptr<ActiveAppearanceModel> model;
  //culib::SimilarityConfig config;

public:
  AamMatchFunction(boost::shared_ptr<culib::ICudaImage> image,
      boost::shared_ptr<ActiveAppearanceModel> model);
  virtual ~AamMatchFunction(void);

  virtual double eval(const DomainType& parameter);
};

}

}

#endif /* GAPPUTILSCV_AAMMATCHFUNCTION_H_ */
