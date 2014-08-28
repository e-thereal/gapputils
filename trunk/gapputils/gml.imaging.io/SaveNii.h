/*
 * SaveNii.h
 *
 *  Created on: Aug 27, 2014
 *      Author: tombr
 */

#ifndef GML_SAVENII_H_
#define GML_SAVENII_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class SaveNii : public DefaultWorkflowElement<SaveNii> {

  typedef std::vector<char> data_t;

  InitReflectableClass(SaveNii)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Header, boost::shared_ptr<data_t>)
  Property(Filename, std::string)
  Property(OutpuName, std::string)

public:
  SaveNii();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_SAVENII_H_ */
