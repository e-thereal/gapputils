/*
 * Normalize.h
 *
 *  Created on: Dec 10, 2013
 *      Author: tombr
 */

#ifndef GML_NORMALIZE_H_
#define GML_NORMALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Normalize : public DefaultWorkflowElement<Normalize> {

  typedef double value_t;
  typedef std::vector<value_t> data_t;

  InitReflectableClass(Normalize)

  Property(Inputs, boost::shared_ptr<data_t>)
  Property(Mean, value_t)
  Property(Stddev, value_t)
  Property(Outputs, boost::shared_ptr<data_t>)

public:
  Normalize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_NORMALIZE_H_ */
