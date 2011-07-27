/*
 * AamAnalyser.cpp
 *
 *  Created on: Jul 26, 2011
 *      Author: tombr
 */

#include "AamAnalyser.h"

#include <capputils/EnumerableAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/Xmlizer.h>

#include <gapputils/HideAttribute.h>

#include <algorithm>
#include <valarray>

#include "AamMatchFunction.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

class AamPlotResult : public capputils::reflection::ReflectableClass {

  InitReflectableClass(AamPlotResult)

  Property(Step, double)
  Property(X, vector<double>)
  Property(Y, double)

public:
  AamPlotResult() : _Step(0), _Y(0) { }
};

BeginPropertyDefinitions(AamPlotResult)
  DefineProperty(Step)
  DefineProperty(X, Enumerable<vector<double>, false>())
  DefineProperty(Y)
EndPropertyDefinitions

BeginPropertyDefinitions(AamAnalyser)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ParameterVector, Input("PV"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(PlotDirection, Enumerable<vector<double>, false>(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(From, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(To, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(StepCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(XmlName, Output("Xml"), Filename("XML file (*.xml)"), NotEqual<std::string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

AamAnalyser::AamAnalyser() : _From(-1), _To(1), _StepCount(10), _XmlName("out.xml"), data(0) {
  WfeUpdateTimestamp
  setLabel("AamAnalyser");

  Changed.connect(capputils::EventHandler<AamAnalyser>(this, &AamAnalyser::changedHandler));
}

AamAnalyser::~AamAnalyser() {
  if (data)
    delete data;
}

void AamAnalyser::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

TiXmlNode* AamAnalyser::createPlot(bool inReferenceFrame, AamMatchFunction::SimilarityMeasure measure) const {
  AamMatchFunction objective(getImage(), getActiveAppearanceModel(), inReferenceFrame, measure);

  const int count = getParameterVector()->size();
  vector<double> parameter(count);

  TiXmlNode* plot = new TiXmlElement("Plot");
  AamPlotResult result;

  for (int iMode = 0; iMode < count; ++iMode) {
    copy(getParameterVector()->begin(), getParameterVector()->end(), parameter.begin());
    TiXmlNode* modePlot = new TiXmlElement("ModePlot");
    for (int iStep = 0; iStep < getStepCount(); ++iStep) {
      double dStep = (1. - (double)iStep / (getStepCount()-1)) * getFrom() + (double)iStep / (getStepCount()-1) * getTo();

      //for (int i = 0; i < count; ++i)
      //  parameter[i] = dStep * getPlotDirection()[i] + getParameterVector()->at(i);
      parameter[iMode] = dStep + getParameterVector()->at(iMode);

      result.setStep(dStep);
      result.setX(parameter);
      result.setY(objective.eval(parameter));
      modePlot->LinkEndChild(capputils::Xmlizer::CreateXml(result));
    }
    plot->LinkEndChild(modePlot);
  }
  return plot;
}

void AamAnalyser::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamAnalyser();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getImage() || !getParameterVector() ||
      getParameterVector()->size() != getPlotDirection().size())
  {
    return;
  }


  TiXmlNode* root = new TiXmlElement("Plots");
  TiXmlElement* plot;

  plot = (TiXmlElement*)createPlot(true, AamMatchFunction::SSD);
  plot->SetAttribute("label", "SSD in reference frame");
  root->LinkEndChild(plot);
  if (monitor)
    monitor->reportProgress(100/4);

  plot = (TiXmlElement*)createPlot(true, AamMatchFunction::MI);
  plot->SetAttribute("label", "MI in reference frame");
  root->LinkEndChild(plot);
  if (monitor)
    monitor->reportProgress(200/4);

  plot = (TiXmlElement*)createPlot(false, AamMatchFunction::SSD);
  plot->SetAttribute("label", "SSD in image frame");
  root->LinkEndChild(plot);
  if (monitor)
    monitor->reportProgress(300/4);

  plot = (TiXmlElement*)createPlot(false, AamMatchFunction::MI);
  plot->SetAttribute("label", "MI in image frame");
  root->LinkEndChild(plot);
  if (monitor)
    monitor->reportProgress(400/4);

  capputils::Xmlizer::ToFile(getXmlName(), root);
}

void AamAnalyser::writeResults() {
  if (!data)
    return;
  setXmlName(getXmlName());
}

}

}
