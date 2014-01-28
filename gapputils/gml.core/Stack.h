/*
 * Stack.h
 *
 *  Created on: Jan 22, 2014
 *      Author: tombr
 */

#ifndef GML_STACK_H_
#define GML_STACK_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Stack : public DefaultWorkflowElement<Stack> {

  InitReflectableClass(Stack)

  typedef double value_t;
  typedef std::vector<value_t> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;
  typedef std::vector<boost::shared_ptr<v_data_t> > vv_data_t;

  Property(Inputs, boost::shared_ptr<vv_data_t>)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Stack();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_STACK_H_ */
