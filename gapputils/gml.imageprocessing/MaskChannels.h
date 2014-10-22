/*
 * MaskChannels.h
 *
 *  Created on: 2013-05-20
 *      Author: tombr
 */

#ifndef GML_MASKCHANNELS_H_
#define GML_MASKCHANNELS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class MaskChannels : public DefaultWorkflowElement<MaskChannels> {

  typedef gapputils::host_tensor_t tensor_t;
  typedef gapputils::v_host_tensor_t v_tensor_t;

  InitReflectableClass(MaskChannels)

  Property(Inputs, boost::shared_ptr<v_tensor_t>)
  Property(ChannelMask, boost::shared_ptr<std::vector<double> >)
  Property(Outputs, boost::shared_ptr<v_tensor_t>)

public:
  MaskChannels();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_MASKCHANNELS_H_ */
