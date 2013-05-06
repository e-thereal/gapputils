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

#include "Model.h"

namespace gml {

namespace convrbm4d {

class TransposeTensorsAndChannels : public DefaultWorkflowElement<TransposeTensorsAndChannels> {

  typedef Model::tensor_t tensor_t;

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

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_TRANSPOSETENSORSANDCHANNELS_H_ */
