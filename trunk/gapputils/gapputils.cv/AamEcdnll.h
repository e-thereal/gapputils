/*
 * AamEcdnll.h
 *
 *  Created on: Jul 20, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMECDNLL_H_
#define GAPPUTILSCV_AAMECDNLL_H_

#include <optlib/IMultiDimensionOptimizer.h>

#include <boost/shared_ptr.hpp>

#include "ActiveAppearanceModel.h"

namespace gapputils {

namespace cv {

/// Calculates the expected complete data negative log likelihood for an active appearance model
class AamEcdnll : public virtual optlib::IFunction<optlib::IMultiDimensionOptimizer::DomainType> {

private:
  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > trainingSet;
  boost::shared_ptr<ActiveAppearanceModel> oldModel;
  boost::shared_ptr<ActiveAppearanceModel> newModel;
  float variance, lambda;
  std::vector<float> sigma;

public:
  AamEcdnll(boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > trainingSet,
      boost::shared_ptr<ActiveAppearanceModel> oldModel,
      float variance, float lambda, std::vector<float>& sigma);
  virtual ~AamEcdnll();

  virtual double eval(const DomainType& parameter);

  boost::shared_ptr<ActiveAppearanceModel> updateModel(const DomainType& parameter);
};

}

}

#endif /* GAPPUTILSCV_AAMECDNLL_H_ */
