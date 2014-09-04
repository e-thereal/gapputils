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

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(FlattenMode, OneVectorPerTensor, SingleVector);

class FlattenTensors : public DefaultWorkflowElement<FlattenTensors> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  InitReflectableClass(FlattenTensors)

  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Tensors, boost::shared_ptr<v_tensor_t>)
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

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_FLATTENTENSORS_H_ */
