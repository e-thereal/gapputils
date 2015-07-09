/*
 * Initialize.h
 *
 *  Created on: Apr 14, 2015
 *      Author: tombr
 */

#ifndef GML_INITIALIZE_H_
#define GML_INITIALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace encoder {

CapputilsEnumerator(ShortcutType, NoShortcut, BottomUp, TopDown);

class Initialize : public DefaultWorkflowElement<Initialize> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  InitReflectableClass(Initialize)

  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(Mask, boost::shared_ptr<host_tensor_t>)
  Property(FilterWidths, std::vector<int>)
  Property(FilterHeights, std::vector<int>)
  Property(FilterDepths, std::vector<int>)
  Property(FilterCounts, std::vector<int>)
  Property(WeightSparsity, double)
  Property(EncodingWeights, std::vector<double>)
  Property(DecodingWeights, std::vector<double>)
  Property(ShortcutWeights, std::vector<double>)
  Property(StrideWidths, std::vector<int>)
  Property(StrideHeights, std::vector<int>)
  Property(StrideDepths, std::vector<int>)
  Property(PoolingWidths, std::vector<int>)
  Property(PoolingHeights, std::vector<int>)
  Property(PoolingDepths, std::vector<int>)
//  Property(HiddenUnitCounts, std::vector<int>)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(NormalizeInputs, bool)
  Property(Shortcuts, ShortcutType)
  Property(Model, boost::shared_ptr<model_t>)

public:
  Initialize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_INITIALIZE_H_ */
