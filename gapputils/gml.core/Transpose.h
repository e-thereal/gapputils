/*
 * Transpose.h
 *
 *  Created on: Apr 23, 2015
 *      Author: tombr
 */

#ifndef GML_TRANSPOSE_H_
#define GML_TRANSPOSE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Transpose : public DefaultWorkflowElement<Transpose> {

  typedef double value_t;
  typedef std::vector<value_t> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Transpose)

  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Transpose();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_TRANSPOSE_H_ */
