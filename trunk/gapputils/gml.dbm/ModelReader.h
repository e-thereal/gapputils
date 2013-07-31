/*
 * ModelReader.h
 *
 *  Created on: Jul 09, 2013
 *      Author: tombr
 */

#ifndef GML_MODELREADER_H_
#define GML_MODELREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbm {

class ModelReader : public DefaultWorkflowElement<ModelReader> {
  InitReflectableClass(ModelReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<Model>)
  Property(FilterWidth, std::vector<int>)
  Property(FilterHeight, std::vector<int>)
  Property(FilterDepth, std::vector<int>)
  Property(ChannelCount, std::vector<int>)
  Property(FilterCount, std::vector<int>)
  Property(HiddenCount, std::vector<int>)

public:
  ModelReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_MODELREADER_H_ */
