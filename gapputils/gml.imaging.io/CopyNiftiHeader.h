/*
 * CopyNiftiHeader.h
 *
 *  Created on: Nov 13, 2013
 *      Author: tombr
 */

#ifndef GML_COPYNIFTIHEADER_H_
#define GML_COPYNIFTIHEADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class CopyNiftiHeader : public DefaultWorkflowElement<CopyNiftiHeader> {

  InitReflectableClass(CopyNiftiHeader)

  Property(InputName, std::string)
  Property(HeaderName, std::string)
  Property(OutputName, std::string)
  Property(Output, std::string)

public:
  CopyNiftiHeader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_COPYNIFTIHEADER_H_ */
