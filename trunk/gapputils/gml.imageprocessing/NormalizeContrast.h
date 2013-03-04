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

  InitReflectableClass(NormalizeContrast)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  NormalizeContrast();

protected:
  void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_NORMALIZECONTRAST_H_ */
