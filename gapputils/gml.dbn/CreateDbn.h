/*
 * CreateDbn.h
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#ifndef GML_CREATEDBN_H_
#define GML_CREATEDBN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

class CreateDbn : public DefaultWorkflowElement<CreateDbn> {

  InitReflectableClass(CreateDbn)

  Property(CrbmModels, boost::shared_ptr<std::vector<boost::shared_ptr<crbm_t> > >)
  Property(RbmModels, boost::shared_ptr<std::vector<boost::shared_ptr<rbm_t> > >)
  Property(DbnModel, boost::shared_ptr<dbn_t>)

public:
  CreateDbn();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* GML_CREATEDBN_H_ */
