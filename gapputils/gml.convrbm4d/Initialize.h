/*
 * Initialize.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef GML_INITIALIZE_H_
#define GML_INITIALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

struct InitializeChecker { InitializeChecker(); };

class Initialize : public DefaultWorkflowElement<Initialize> {
public:
  typedef model_t::value_t value_t;
  typedef model_t::host_tensor_t host_tensor_t;
  typedef model_t::v_host_tensor_t v_host_tensor_t;

  friend class InitializeChecker;

  InitReflectableClass(Initialize)

  Property(Tensors, boost::shared_ptr<v_host_tensor_t>)
  Property(Mask, boost::shared_ptr<host_tensor_t>)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(FilterCount, int)
  Property(StrideWidth, int)
  Property(StrideHeight, int)
  Property(StrideDepth, int)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(PoolingWidth, int)
  Property(PoolingHeight, int)
  Property(PoolingDepth, int)
  Property(WeightMean, double)
  Property(WeightStddev, double)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)

  Property(Model, boost::shared_ptr<model_t>)

public:
  Initialize();
  virtual ~Initialize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_INITIALIZE_H_ */
