/*
 * TransposeTensorsAndChannels.h
 *
 *  Created on: Apr 25, 2013
 *      Author: tombr
 */

#ifndef GML_TRANSPOSETENSORSANDCHANNELS_H_
#define GML_TRANSPOSETENSORSANDCHANNELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace core {

class TransposeTensorsAndChannels : public DefaultWorkflowElement<TransposeTensorsAndChannels> {

  typedef tbblas::tensor<float, 4> tensor_t;

  InitReflectableClass(TransposeTensorsAndChannels)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(InputTensor, boost::shared_ptr<tensor_t>)
  Property(OutputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(OutputTensor, boost::shared_ptr<tensor_t>)

public:
  TransposeTensorsAndChannels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_TRANSPOSETENSORSANDCHANNELS_H_ */
