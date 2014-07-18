/*
 * ModelWriter.h
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef GML_MODELWRITER_H_
#define GML_MODELWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

class ModelWriter : public DefaultWorkflowElement<ModelWriter> {

  InitReflectableClass(ModelWriter)

  Property(Model, boost::shared_ptr<dbn_t>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  ModelWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* GML_MODELWRITER_H_ */
