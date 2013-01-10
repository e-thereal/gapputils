/*
 * ModelReader.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

class ModelReader : public DefaultWorkflowElement<ModelReader> {
  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<Model>)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(ChannelCount, int)
  Property(FilterCount, int)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* GML_MODELREADER_H_ */