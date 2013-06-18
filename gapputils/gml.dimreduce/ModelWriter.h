/*
 * ModelWriter.h
 *
 *  Created on: Jun 18, 2013
 *      Author: tombr
 */

#ifndef MODELWRITER_H_
#define MODELWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dimreduce {

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

} /* namespace dimreduce */
} /* namespace gml */
#endif /* MODELWRITER_H_ */
