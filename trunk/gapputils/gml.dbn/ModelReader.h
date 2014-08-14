/*
 * ModelReader.h
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

class ModelReader : public DefaultWorkflowElement<ModelReader> {

  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<dbn_t>)
  Property(ConvolutionalLayers, int)
  Property(DenseLayers, int)
  Property(FilterCounts, std::vector<int>)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */
} /* namespace gml */
#endif /* GML_MODELREADER_H_ */
