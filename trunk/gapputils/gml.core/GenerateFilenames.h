/*
 * GenerateFilenames.h
 *
 *  Created on: Feb 3, 2014
 *      Author: tombr
 */

#ifndef GML_GENERATEFILENAMES_H_
#define GML_GENERATEFILENAMES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class GenerateFilenames : public DefaultWorkflowElement<GenerateFilenames>  {

  InitReflectableClass(GenerateFilenames)

  Property(Pattern, std::string)
  Property(Counts, std::vector<int>)
  Property(Order, std::vector<int>)
  Property(OutputNames, std::vector<std::string>)

public:
  GenerateFilenames();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_GENERATEFILENAMES_H_ */
