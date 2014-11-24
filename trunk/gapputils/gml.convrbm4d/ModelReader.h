/*
 * ModelReader.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

class ModelReader : public DefaultWorkflowElement<ModelReader> {
  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<model_t>)
  Property(TensorWidth, int)
  Property(TensorHeight, int)
  Property(TensorDepth, int)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(ChannelCount, int)
  Property(FilterCount, int)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(PoolingSize, model_t::host_tensor_t::dim_t)
  Property(Mean, double)
  Property(Stddev, double)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* GML_MODELREADER_H_ */
