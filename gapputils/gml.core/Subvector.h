/*
 * Subvector.h
 *
 *  Created on: Jan 29, 2014
 *      Author: tombr
 */

#ifndef GML_SUBVECTOR_H_
#define GML_SUBVECTOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class Subvector : public DefaultWorkflowElement<Subvector> {

  typedef double value_t;
  typedef std::vector<value_t> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Subvector)

  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(StartIndex, int)
  Property(Length, int)
  Property(Outputs, boost::shared_ptr<v_data_t>)

public:
  Subvector();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_SUBVECTOR_H_ */
