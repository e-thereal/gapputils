/*
 * FgrbmModel.h
 *
 *  Created on: Nov 28, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_FGRBMMODEL_H_
#define GAPPUTILS_ML_FGRBMMODEL_H_

#include <capputils/ReflectableClass.h>

#include <tbblas/device_matrix.hpp>
#include <tbblas/device_vector.hpp>

namespace gapputils {

namespace ml {

class FgrbmModel : public capputils::reflection::ReflectableClass {

  InitReflectableClass(FgrbmModel)

  Property(VisibleBiases, boost::shared_ptr<tbblas::device_vector<float> >)
  Property(HiddenBiases, boost::shared_ptr<tbblas::device_vector<float> >)
  Property(VisibleWeights, boost::shared_ptr<tbblas::device_matrix<float> >)
  Property(HiddenWeights, boost::shared_ptr<tbblas::device_matrix<float> >)
  Property(ConditionalWeights, boost::shared_ptr<tbblas::device_matrix<float> >)
  Property(VisibleMean, float)
  Property(VisibleStd, float)

public:
  FgrbmModel();
  virtual ~FgrbmModel();
};

}

}

#endif /* GAPPUTILS_ML_FGRBMMODEL_H_ */
