/*
 * AugmentDataset.h
 *
 *  Created on: Jan 23, 2015
 *      Author: tombr
 */

#ifndef GML_AUGMENTDATASET_H_
#define GML_AUGMENTDATASET_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

namespace gml {

namespace imageprocessing {

class AugmentDataset : public DefaultWorkflowElement<AugmentDataset> {

  InitReflectableClass(AugmentDataset)

  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(ContrastSd, double)
  Property(BrightnessSd, double)
  Property(GammaSd, double)
  Property(SampleCount, int)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  AugmentDataset();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_AUGMENTDATASET_H_ */
