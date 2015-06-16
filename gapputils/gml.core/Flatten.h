/*
 * Flatten.h
 *
 *  Created on: Jun 1, 2015
 *      Author: tombr
 */

#ifndef GML_FLATTEN_H_
#define GML_FLATTEN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Flatten : public DefaultWorkflowElement<Flatten> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Flatten)

  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(Outputs, boost::shared_ptr<data_t>)

public:
  Flatten();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_FLATTEN_H_ */
