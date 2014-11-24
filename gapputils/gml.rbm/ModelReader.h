/*
 * ModelReader.h
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace rbm {

class ModelReader : public DefaultWorkflowElement<ModelReader> {

  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<model_t>)
  Property(FloatModel, bool)
  Property(VisibleCount, int)
  Property(HiddenCount, int)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_MODELREADER_H_ */
