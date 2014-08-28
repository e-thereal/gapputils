/*
 * SaveImage.h
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#ifndef GML_SAVEIMAGE_H
#define GML_SAVEIMAGE_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class SaveImage : public DefaultWorkflowElement<SaveImage> {

  InitReflectableClass(SaveImage)

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
  SaveImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
  void changedHandler(capputils::ObservableClass* sender, int eventId);

private:
  void saveImage(image_t& image, const std::string& filename) const;
};

}

}

}

#endif /* GML_SAVEIMAGE_H */
