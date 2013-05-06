/*
 * MakeTensors.h
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#ifndef GML_MAKETENSORS_H_
#define GML_MAKETENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class MakeTensors : public DefaultWorkflowElement<MakeTensors> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(MakeTensors)

  Property(Vectors, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(ChannelCount, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Tensor, boost::shared_ptr<tensor_t>)

public:
  MakeTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_MAKETENSORS_H_ */
