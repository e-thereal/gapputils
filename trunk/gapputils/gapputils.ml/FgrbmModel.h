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

  int dummyEntry;   ///< Brings the struct into 8 byte alignment. This is necessary in order to ensure
                    ///< compatibility between gcc and nvcc

  Property(VisibleBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(HiddenBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(VisibleWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(HiddenWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(ConditionalWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(VisibleMean, double)
  Property(VisibleStd, double)

public:
  FgrbmModel();
  virtual ~FgrbmModel();
};

}

}

#endif /* GAPPUTILS_ML_FGRBMMODEL_H_ */
