/*
 * OldMifReader.h
 *
 *  Created on: Jul 5, 2013
 *      Author: tombr
 */

#ifndef GML_OLDMIFREADER_H_
#define GML_OLDMIFREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class OldMifReader : public DefaultWorkflowElement<OldMifReader> {

  InitReflectableClass(OldMifReader)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)

public:
  OldMifReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_OLDMIFREADER_H_ */
