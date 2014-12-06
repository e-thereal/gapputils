/*
 * OpenModel.h
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#ifndef GML_OPENMODEL_H_
#define GML_OPENMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace jcnn {

class OpenModel : public DefaultWorkflowElement<OpenModel> {

  InitReflectableClass(OpenModel)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<model_t>)
  Property(LeftInputSize, model_t::dim_t)
  Property(LeftFilterCounts, std::vector<int>)
  Property(LeftHiddenCounts, std::vector<int>)
  Property(LeftConvolutionType, tbblas::deeplearn::convolution_type)

  Property(RightInputSize, model_t::dim_t)
  Property(RightFilterCounts, std::vector<int>)
  Property(RightHiddenCounts, std::vector<int>)
  Property(RightConvolutionType, tbblas::deeplearn::convolution_type)

  Property(JointHiddenCounts, std::vector<int>)

  Property(OutputCount, int)
  Property(HiddenActivationFunction, tbblas::deeplearn::activation_function)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)

public:
  OpenModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace jcnn */

} /* namespace gml */

#endif /* GML_OPENMODEL_H_ */
