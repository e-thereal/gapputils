/*
 * AamAnalyser.h
 *
 *  Created on: Jul 26, 2011
 *      Author: tombr
 */

#ifndef GAPPUTILSCV_AAMANALYSER_H_
#define GAPPUTILSCV_AAMANALYSER_H_

#include <gapputils/WorkflowElement.h>

#include "ActiveAppearanceModel.h"
#include "AamMatchFunction.h"

#include <tinyxml/tinyxml.h>

namespace gapputils {

namespace cv {

class AamAnalyser : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(AamAnalyser)

  Property(ActiveAppearanceModel, boost::shared_ptr<ActiveAppearanceModel>)
  Property(Image, boost::shared_ptr<culib::ICudaImage>)
  Property(ParameterVector, boost::shared_ptr<std::vector<float> >)
  Property(PlotDirection, std::vector<double>)
  Property(From, double)
  Property(To, double)
  Property(StepCount, int)
  Property(XmlName, std::string)

private:
  mutable AamAnalyser* data;

public:
  AamAnalyser();
  virtual ~AamAnalyser();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

private:
  TiXmlNode* createPlot(bool inReferenceFrame, AamMatchFunction::SimilarityMeasure measure) const;
};

}

}


#endif /* GAPPUTILSCV_AAMANALYSER_H_ */
