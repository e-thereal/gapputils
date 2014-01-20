/*
 * RandomizeFileList.h
 *
 *  Created on: Jan 15, 2014
 *      Author: tombr
 */

#ifndef GML_RANDOMIZEFILELIST_H_
#define GML_RANDOMIZEFILELIST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class RandomizeFileList : public DefaultWorkflowElement<RandomizeFileList> {

  InitReflectableClass(RandomizeFileList)

  Property(InList, std::vector<std::string>)
  Property(OutList, std::vector<std::string>)

public:
  RandomizeFileList();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace gml */

#endif /* GML_RANDOMIZEFILELIST_H_ */
