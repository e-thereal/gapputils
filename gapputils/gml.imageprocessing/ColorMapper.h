/*
 * ColorMapper.h
 *
 *  Created on: Jan 22, 2013
 *      Author: tombr
 */

#ifndef GML_COLORMAPPER_H_
#define GML_COLORMAPPER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(ColorMap, Greyscale, HeatMap1, HeatMap2);

class ColorMapper : public DefaultWorkflowElement<ColorMapper> {

  InitReflectableClass(ColorMapper)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(ColorMap, ColorMap)
  Property(MinimumIntensity, double)
  Property(MaximumIntensity, double)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  ColorMapper();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_COLORMAPPER_H_ */
