/*
 * AdjustIntensities.h
 *
 *  Created on: Oct 26, 2012
 *      Author: tombr
 */

#ifndef GML_IMAGEPROCESSING_ADJUSTINTENSITIES_H_
#define GML_IMAGEPROCESSING_ADJUSTINTENSITIES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class AdjustIntensities : public DefaultWorkflowElement<AdjustIntensities> {
  InitReflectableClass(AdjustIntensities)

  Property(Input, boost::shared_ptr<image_t>)
  Property(InputIntensities, boost::shared_ptr<std::vector<double> >)
  Property(OutputIntensities, boost::shared_ptr<std::vector<double> >)
  Property(Output, boost::shared_ptr<image_t>)

public:
  AdjustIntensities();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

} /* namespace gml */
#endif /* GML_IMAGEPROCESSING_ADJUSTINTENSITIES_H_ */
