/*
 * Histogram.h
 *
 *  Created on: Oct 24, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_CV_HISTOGRAM_H_
#define GAPPUTLIS_CV_HISTOGRAM_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

namespace gapputils {
namespace cv {

class Histogram : public workflow::DefaultWorkflowElement<Histogram> {

  InitReflectableClass(Histogram)

  Property(Image, boost::shared_ptr<image_t>)
  Property(BinCount, int)
  Property(HistogramBinWidth, int)
  Property(HistogramHeight, int)
  Property(AverageHeight, int)
  Property(Foreground, float)
  Property(Background, float)
  Property(ModeColor, float)
  Property(ModeRadius, int)
  Property(MinMode, int)
  Property(ModeCount, int)
  Property(SmoothingRadius, int)
  Property(Histogram, boost::shared_ptr<image_t>)
  Property(Modes, boost::shared_ptr<std::vector<double> >)

public:
  Histogram();
  virtual ~Histogram();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

} /* namespace cv */
} /* namespace gapputils */
#endif /* GAPPUTLIS_CV_HISTOGRAM_H_ */
