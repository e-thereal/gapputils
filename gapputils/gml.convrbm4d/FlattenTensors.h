/*
 * FlattenTensors.h
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#ifndef GML_FLATTENTENSORS_H_
#define GML_FLATTENTENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(FlattenMode, OneVectorPerTensor, SingleVector);

class FlattenTensors : public DefaultWorkflowElement<FlattenTensors> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(FlattenTensors)

  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Mode, FlattenMode)
  Property(Vectors, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(Vector, boost::shared_ptr<std::vector<double> >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(ChannelCount, int)
  Property(Count, int)

public:
  FlattenTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_FLATTENTENSORS_H_ */
