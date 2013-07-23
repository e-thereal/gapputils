/*
 * CreateDbm.h
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#ifndef GML_CREATEDBM_H_
#define GML_CREATEDBM_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbm {

struct CreateDbmChecker { CreateDbmChecker(); };

class CreateDbm : public DefaultWorkflowElement<CreateDbm> {

  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;
  typedef Model::v_tensor_t v_host_tensor_t;
  typedef Model::vv_tensor_t vv_host_tensor_t;
  typedef Model::matrix_t host_matrix_t;
  typedef Model::v_matrix_t v_host_matrix_t;

  friend class CreateDbmChecker;

  InitReflectableClass(CreateDbm)

  Property(Dataset, boost::shared_ptr<v_host_tensor_t>)
  Property(CrbmModels, boost::shared_ptr<std::vector<boost::shared_ptr<gml::convrbm4d::Model> > >)
  Property(RbmModels, boost::shared_ptr<std::vector<boost::shared_ptr<gml::rbm::Model> > >)
  Property(DbmModel, boost::shared_ptr<Model>)

public:
  CreateDbm();

protected:
  virtual void update(IProgressMonitor* monitor) const;

};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_CREATEDBM_H_ */
