/*
 * MnistReader.h
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#ifndef GML_MNISTREADER_H_
#define GML_MNISTREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class MnistReader : public DefaultWorkflowElement<MnistReader> {

  InitReflectableClass(MnistReader)

  Property(ImageFile, std::string)
  Property(LabelFile, std::string)
  Property(MaxImageCount, int)
  Property(SelectedDigits, std::vector<int>)
  Property(MakeBinary, bool)
  Property(Images, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(Labels, boost::shared_ptr<std::vector<double> >)
  Property(ImageCount, int)
  Property(Width, int)
  Property(Height, int)

public:
  MnistReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_MNISTREADER_H_ */
