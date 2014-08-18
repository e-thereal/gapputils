/*
 * ConvertRbms.h
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#ifndef GML_CONVERTRBMS_H_
#define GML_CONVERTRBMS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

#include <tbblas/deeplearn/rbm_model.hpp>

namespace gml {

namespace nn {

// TODO: implement

class ConvertRbms : public DefaultWorkflowElement<ConvertRbms> {

  typedef tbblas::deeplearn::rbm_model<double> rbm_t;
  typedef std::vector<boost::shared_ptr<rbm_t> > v_rbm_t;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(ConvertRbms)

  Property(TrainingSet, boost::shared_ptr<v_data_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(Rbms, boost::shared_ptr<v_rbm_t>)
  Property(InitialWeights, double)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(Model, boost::shared_ptr<model_t>)

public:
  ConvertRbms();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_CONVERTRBMS_H_ */
