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

#include <tbblas/deeplearn/conv_rbm_model.hpp>
#include <tbblas/deeplearn/rbm_model.hpp>

namespace gml {

namespace encoder {

class ConvertRbms : public DefaultWorkflowElement<ConvertRbms> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef tbblas::deeplearn::rbm_model<double> rbm_t;
  typedef std::vector<boost::shared_ptr<rbm_t> > v_rbm_t;

  typedef tbblas::deeplearn::conv_rbm_model<float, 4> crbm_t;
  typedef std::vector<boost::shared_ptr<crbm_t> > v_crbm_t;

  InitReflectableClass(ConvertRbms)

  Property(Crbms, boost::shared_ptr<v_crbm_t>)
  Property(Rbms, boost::shared_ptr<v_rbm_t>)
  Property(FirstInputChannel, int)
  Property(InputChannelCount, int)
  Property(FirstOutputChannel, int)
  Property(OutputChannelCount, int)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(InsertShortcuts, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  ConvertRbms();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_CONVERTRBMS_H_ */
