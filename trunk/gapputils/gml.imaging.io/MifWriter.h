/*
 * MifWriter.h
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#ifndef GML_MIFWRITER_H
#define GML_MIFWRITER_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class MifWriter : public DefaultWorkflowElement<MifWriter> {

  InitReflectableClass(MifWriter)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(MifName, std::string)
  Property(MinValue, double)
  Property(MaxValue, double)
  Property(MaximumIntensity, int)
  Property(AutoScale, bool)
  Property(OutputName, std::string)

public:
  MifWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_MIFWRITER_H */
