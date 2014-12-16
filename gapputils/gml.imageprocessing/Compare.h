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

#include "SimilarityMeasure.h"

namespace gml {

namespace imageprocessing {

struct CompareChecker { CompareChecker(); };

class MeasureParameters : public capputils::reflection::ReflectableClass,
                          public ObservableClass
{
  InitReflectableClass(MeasureParameters)
};

class NoMeasureParameters : public MeasureParameters {
  InitReflectableClass(NoMeasureParameters)
};

class SsimParameters : public MeasureParameters {

  friend struct CompareChecker;

  InitReflectableClass(SsimParameters)

  Property(WindowWidth, int)
  Property(WindowHeight, int)
  Property(WindowDepth, int)

public:
  SsimParameters();
};

class Compare : public DefaultWorkflowElement<Compare> {

  friend struct CompareChecker;

  InitReflectableClass(Compare)

  Property(Image1, boost::shared_ptr<image_t>)
  Property(Image2, boost::shared_ptr<image_t>)
  Property(Measure, SimilarityMeasure)
  Property(Parameters, boost::shared_ptr<MeasureParameters>)
  Property(Value, double)

  static int measureId;

public:
  Compare();

protected:
  virtual void update(IProgressMonitor* monitor) const;

  void changedHandler(ObservableClass* sender, int eventId);
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* COMPARE_H_ */
