/*
 * Aggregate.h
 *
 *  Created on: Jan 23, 2013
 *      Author: tombr
 */

#ifndef GML_AGGREGATE_H_
#define GML_AGGREGATE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "AggregatorOperation.h"

namespace gml {

namespace core {

class Aggregate : public DefaultWorkflowElement<Aggregate> {

  InitReflectableClass(Aggregate)

  Property(Data, boost::shared_ptr<std::vector<double> >)
  Property(Operation, AggregatorOperation)
  Property(Value, double)

public:
  Aggregate();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_AGGREGATE_H_ */
