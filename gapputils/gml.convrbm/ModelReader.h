/*
 * ModelReader.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef MODELREADER_H_
#define MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm {

class ModelReader : public DefaultWorkflowElement<ModelReader> {
  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<Model>)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterDepth, int)
  Property(FilterCount, int)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)

public:
  ModelReader();
  virtual ~ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm */
} /* namespace gml */
#endif /* MODELREADER_H_ */
