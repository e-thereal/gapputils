/*
 * InitializePatch.h
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#ifndef GML_INITIALIZEPATCH_H_
#define GML_INITIALIZEPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

class InitializePatch : public DefaultWorkflowElement<InitializePatch> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef tbblas::tensor<value_t, dimCount> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(InitializePatch)

  Property(TrainingSet, boost::shared_ptr<v_tensor_t>)
  Property(Labels, boost::shared_ptr<v_tensor_t>)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(FilterWidths, std::vector<int>)
  Property(FilterHeights, std::vector<int>)
  Property(FilterDepths, std::vector<int>)
  Property(FilterCounts, std::vector<int>)
  Property(PoolingWidths, std::vector<int>)
  Property(PoolingHeights, std::vector<int>)
  Property(PoolingDepths, std::vector<int>)
  Property(HiddenUnitCounts, std::vector<int>)
  Property(InitialWeights, double)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(NormalizeInputs, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  InitializePatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cnn */

} /* namespace gml */

#endif /* GML_INITIALIZEPATCH_H_ */
