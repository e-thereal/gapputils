/*
 * Pooling.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef POOLING_H_
#define POOLING_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "PoolingMethod.h"
#include "CodingDirection.h"
#include "Model.h"

namespace gml {
namespace convrbm4d {

class Pooling : public DefaultWorkflowElement<Pooling> {

  typedef model_t::host_tensor_t tensor_t;

  InitReflectableClass(Pooling)

  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(BlockWidth, int)
  Property(BlockHeight, int)
  Property(BlockDepth, int)
  Property(Method, PoolingMethod)
  Property(Direction, CodingDirection)
  Property(IntensityCorrection, double)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(IntensityRatio, double)

public:
  Pooling();
  virtual ~Pooling();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */

#endif /* POOLING_H_ */
