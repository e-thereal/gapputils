/*
 * MaskChannels.h
 *
 *  Created on: 2013-05-20
 *      Author: tombr
 */

#ifndef GML_MASKCHANNELS_H_
#define GML_MASKCHANNELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class MaskChannels : public DefaultWorkflowElement<MaskChannels> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(MaskChannels)

  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(ChannelMask, boost::shared_ptr<std::vector<double> >)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)

public:
  MaskChannels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_MASKCHANNELS_H_ */
