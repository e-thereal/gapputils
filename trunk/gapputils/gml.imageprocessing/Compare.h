/*
 * Compare.h
 *
 *  Created on: Jan 23, 2013
 *      Author: tombr
 */

#ifndef GMLCOMPARE_H_
#define GMLCOMPARE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(SimilarityMeasure, MSE, SSIM)

class Compare : public DefaultWorkflowElement<Compare> {

  InitReflectableClass(Compare)

  Property(Image1, boost::shared_ptr<image_t>)
  Property(Image2, boost::shared_ptr<image_t>)
  Property(Measure, SimilarityMeasure)
  Property(Value, double)

public:
  Compare();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* COMPARE_H_ */
