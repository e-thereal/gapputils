/*
 * CompareTensor.h
 *
 *  Created on: Dec 7, 2014
 *      Author: tombr
 */

#ifndef GML_COMPARETENSOR_H_
#define GML_COMPARETENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include "SimilarityMeasure.h"

namespace gml {

namespace imageprocessing {

class CompareTensor : public DefaultWorkflowElement<CompareTensor> {

  InitReflectableClass(CompareTensor)

  Property(Input, boost::shared_ptr<host_tensor_t>)
  Property(Gold, boost::shared_ptr<host_tensor_t>)
  Property(Measure, SimilarityMeasure)
  Property(Value, double)

public:
  CompareTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_COMPARETENSOR_H_ */
