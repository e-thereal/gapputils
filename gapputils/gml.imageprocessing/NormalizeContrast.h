/*
 * NormalizeContrast.h
 *
 *  Created on: Mar 1, 2013
 *      Author: tombr
 */

#ifndef GML_NORMALIZECONTRAST_H_
#define GML_NORMALIZECONTRAST_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {
namespace imageprocessing {

class NormalizeContrast : public DefaultWorkflowElement<NormalizeContrast> {

  typedef std::vector<boost::shared_ptr<image_t> > v_image_t;

  InitReflectableClass(NormalizeContrast)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(InputImages, boost::shared_ptr<v_image_t>)
  Property(IntensityWindow, std::vector<double>)
  Property(OutputImage, boost::shared_ptr<image_t>)
  Property(OutputImages, boost::shared_ptr<v_image_t>)

private:
  static int intensityId;

public:
  NormalizeContrast();

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_NORMALIZECONTRAST_H_ */
