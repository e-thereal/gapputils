/*
 * Initialize.h
 *
 *  Created on: Dec 03, 2014
 *      Author: tombr
 */

#ifndef GML_INITIALIZE_H_
#define GML_INITIALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace jcnn {

class Initialize : public DefaultWorkflowElement<Initialize> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(Initialize)

  Property(LeftTrainingSet, boost::shared_ptr<v_tensor_t>)
  Property(RightTrainingSet, boost::shared_ptr<v_tensor_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(LeftFilterWidths, std::vector<int>)
  Property(LeftFilterHeights, std::vector<int>)
  Property(LeftFilterDepths, std::vector<int>)
  Property(LeftFilterCounts, std::vector<int>)
  Property(LeftStrideWidths, std::vector<int>)
  Property(LeftStrideHeights, std::vector<int>)
  Property(LeftStrideDepths, std::vector<int>)
  Property(LeftPoolingWidths, std::vector<int>)
  Property(LeftPoolingHeights, std::vector<int>)
  Property(LeftPoolingDepths, std::vector<int>)
  Property(LeftHiddenUnitCounts, std::vector<int>)
  Property(RightFilterWidths, std::vector<int>)
  Property(RightFilterHeights, std::vector<int>)
  Property(RightFilterDepths, std::vector<int>)
  Property(RightFilterCounts, std::vector<int>)
  Property(RightStrideWidths, std::vector<int>)
  Property(RightStrideHeights, std::vector<int>)
  Property(RightStrideDepths, std::vector<int>)
  Property(RightPoolingWidths, std::vector<int>)
  Property(RightPoolingHeights, std::vector<int>)
  Property(RightPoolingDepths, std::vector<int>)
  Property(RightHiddenUnitCounts, std::vector<int>)
  Property(JointHiddenUnitCounts, std::vector<int>)
  Property(InitialWeights, double)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(PoolingMethod, tbblas::deeplearn::pooling_method)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(NormalizeInputs, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  Initialize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace jcnn */

} /* namespace gml */

#endif /* GML_INITIALIZE_H_ */
