/*
 * InitializePatch.h
 *
 *  Created on: Oct 16, 2014
 *      Author: tombr
 */

#ifndef GML_INITIALIZEPATCH_H_
#define GML_INITIALIZEPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

struct InitializePatchChecker { InitializePatchChecker(); };

class InitializePatch : public DefaultWorkflowElement<InitializePatch> {
public:
  typedef model_t::value_t value_t;
  typedef model_t::host_tensor_t host_tensor_t;
  typedef model_t::v_host_tensor_t v_host_tensor_t;

  friend class InitializePatchChecker;

  InitReflectableClass(InitializePatch)

  Property(Tensors, boost::shared_ptr<v_host_tensor_t>)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(FilterCount, int)
  Property(StrideWidth, int)
  Property(StrideHeight, int)
  Property(StrideDepth, int)
  Property(WeightMean, double)
  Property(WeightStddev, double)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(PatchChannels, int)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)
  Property(ConvolutionType, tbblas::deeplearn::convolution_type)

  Property(Model, boost::shared_ptr<model_t>)

public:
  InitializePatch();
  virtual ~InitializePatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_INITIALIZEPATCH_H_ */
