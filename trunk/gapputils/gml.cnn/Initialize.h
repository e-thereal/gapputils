/*
 * Initialize.h
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef GML_INITIALIZE_H_
#define GML_INITIALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

class Initialize : public DefaultWorkflowElement<Initialize> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(Initialize)

  Property(TrainingSet, boost::shared_ptr<v_tensor_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(FilterWidths, std::vector<int>)
  Property(FilterHeights, std::vector<int>)
  Property(FilterDepths, std::vector<int>)
  Property(FilterCounts, std::vector<int>)
  Property(StrideWidths, std::vector<int>)
  Property(StrideHeights, std::vector<int>)
  Property(StrideDepths, std::vector<int>)
  Property(HiddenUnitCounts, std::vector<int>)
  Property(InitialWeights, double)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(NormalizeInputs, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  Initialize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_INITIALIZE_H_ */
