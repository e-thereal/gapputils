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

namespace nn {

class Initialize : public DefaultWorkflowElement<Initialize> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Initialize)

  Property(TrainingSet, boost::shared_ptr<v_data_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(HiddenUnitCounts, std::vector<int>)
  Property(InitialWeights, double)
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
