/*
 * ModelWriter.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_MODELWRITER_H_
#define GML_MODELWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm4d {

class ModelWriter : public DefaultWorkflowElement<ModelWriter> {

  InitReflectableClass(ModelWriter)

  Property(Model, boost::shared_ptr<Model>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  ModelWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* MODELWRITER_H_ */
