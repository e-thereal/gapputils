/*
 * AggregateTensors.h
 *
 *  Created on: Feb 9, 2015
 *      Author: tombr
 */

#ifndef GML_AGGREGATETENSORS_H_
#define GML_AGGREGATETENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

#include "AggregatorFunction.h"

namespace gml {
namespace imageprocessing {

class AggregateTensors : public DefaultWorkflowElement<AggregateTensors> {

  typedef std::vector<boost::shared_ptr<v_host_tensor_t> > vv_host_tensor_t;

  InitReflectableClass(AggregateTensors)

  Property(Inputs, boost::shared_ptr<vv_host_tensor_t>)
  Property(Function, AggregatorFunction)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  AggregateTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_AGGREGATETENSORS_H_ */
