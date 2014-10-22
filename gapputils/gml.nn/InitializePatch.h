/*
 * InitializePatch.h
 *
 *  Created on: Oct 14, 2014
 *      Author: tombr
 */

#ifndef GML_INITIALIZEPATCH_H_
#define GML_INITIALIZEPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

#include "Model.h"

namespace gml {

namespace nn {

class InitializePatch : public DefaultWorkflowElement<InitializePatch> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(InitializePatch)

  Property(TrainingSet, boost::shared_ptr<v_tensor_t>)
  Property(Labels, boost::shared_ptr<v_tensor_t>)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(HiddenUnitCounts, std::vector<int>)
  Property(InitialWeights, double)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(NormalizeInputs, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  InitializePatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_INITIALIZEPATCH_H_ */
