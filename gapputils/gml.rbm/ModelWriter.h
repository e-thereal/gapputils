/*
 * ModelWriter.h
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#ifndef GML_MODELWRITER_H_
#define GML_MODELWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace rbm {

class ModelWriter : public DefaultWorkflowElement<ModelWriter> {

  InitReflectableClass(ModelWriter)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  ModelWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

#endif /* GML_MODELWRITER_H_ */
