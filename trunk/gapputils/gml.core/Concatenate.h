/*
 * Concatenate.h
 *
 *  Created on: Nov 15, 2013
 *      Author: tombr
 */

#ifndef GML_CONCATENATE_H_
#define GML_CONCATENATE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Concatenate : public DefaultWorkflowElement<Concatenate> {

  InitReflectableClass(Concatenate)

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<v_data_t> > >)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Concatenate();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_CONCATENATE_H_ */
