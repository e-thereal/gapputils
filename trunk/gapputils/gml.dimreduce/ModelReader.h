/*
 * ModelReader.h
 *
 *  Created on: Jun 18, 2013
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dimreduce {

class ModelReader : public DefaultWorkflowElement<ModelReader> {

  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<Model>)
  Property(Method, DimensionalityReductionMethod)
//  Property(ManifoldDimensions, int)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dimreduce */

} /* namespace gml */

#endif /* GML_MODELREADER_H_ */
