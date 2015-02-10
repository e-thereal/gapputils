/*
 * Resample.h
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#ifndef GML_RESAMPLE_H_
#define GML_RESAMPLE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imageprocessing {

struct ResampleChecker { ResampleChecker(); };

class Resample : public DefaultWorkflowElement<Resample> {

  typedef tbblas::sequence<int, 3> dim_t;

  InitReflectableClass(Resample)

  friend class ResampleChecker;

  Property(Input, boost::shared_ptr<image_t>)
  Property(Size, dim_t)
  Property(PixelSize, dim_t)
  Property(Output, boost::shared_ptr<image_t>)

public:
  Resample();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_RESAMPLE_H_ */
