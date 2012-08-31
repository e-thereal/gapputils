/*
 * ConvRbmModel.h
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ML_CONVRBMMODEL_H_
#define GAPPUTLIS_ML_CONVRBMMODEL_H_

#include <capputils/ReflectableClass.h>

#include <tbblas/tensor_base.hpp>

namespace gapputils {

namespace ml {

class InitializeConvRbmModel;

class ConvRbmModel : public capputils::reflection::ReflectableClass {

  friend class InitializeConvRbmModel;

public:
  const static unsigned dimCount = 3;
  typedef double value_t;
  typedef tbblas::tensor_base<value_t, dimCount, false> tensor_t;

private:
  InitReflectableClass(ConvRbmModel)

  int dummyEntry;   ///< Brings the struct into 8 byte alignment. This is necessary in order to ensure
                    ///< compatibility between gcc and nvcc

  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(VisibleBias, value_t)
  Property(HiddenBiases, boost::shared_ptr<std::vector<value_t> >)
  Property(Mean, value_t)
  Property(Stddev, value_t)
  Property(PoolingBlockSize, unsigned)
  Property(IsGaussian, bool)

public:
  ConvRbmModel();
  virtual ~ConvRbmModel();

  boost::shared_ptr<ConvRbmModel> clone();
};

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTLIS_ML_CONVRBMMODEL_H_ */
