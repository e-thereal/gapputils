/*
 * ImageWriter.h
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#ifndef GML_IMAGEWRITER_H
#define GML_IMAGEWRITER_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class ImageWriter : public DefaultWorkflowElement<ImageWriter> {

  InitReflectableClass(ImageWriter)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Filename, std::string)
  Property(AutoSave, bool)
  Property(AutoName, std::string)
  Property(AutoSuffix, std::string)
  Property(OutputName, std::string)

private:
  static int imageId;
  int imageNumber;

public:
  ImageWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

}

#endif /* GML_IMAGEWRITER_H */
