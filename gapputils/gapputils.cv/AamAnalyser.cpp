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

#include <capputils/HideAttribute.h>

#include <algorithm>
#include <valarray>

#include "AamMatchFunction.h"

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

class AamPlotResult : public capputils::reflection::ReflectableClass {

  InitReflectableClass(AamPlotResult)

  Property(Step, double)
  Property(XStep, double)
  Property(YStep, double)
  Property(X, vector<double>)
  Property(Y, double)

public:
  AamPlotResult() : _Step(0), _XStep(0), _YStep(0), _Y(0) { }
};

BeginPropertyDefinitions(AamPlotResult)
  DefineProperty(Step)
  DefineProperty(XStep)
  DefineProperty(YStep)
  DefineProperty(X, Enumerable<vector<double>, false>())
  DefineProperty(Y)
EndPropertyDefinitions

BeginPropertyDefinitions(AamAnalyser)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(ActiveAppearanceModel, Input("AAM"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Image, Input("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(FocalShapeParameters, Input("FSP"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(StartShapeParameters, Input("SSP"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(TargetShapeParameters, Input("TSP"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
//  DefineProperty(PlotDirection, Enumerable<vector<double>, false>(), Observe(Id), TimeStamp(Id))
  DefineProperty(From, Observe(Id), TimeStamp(Id))
  DefineProperty(To, Observe(Id), TimeStamp(Id))
  DefineProperty(StepCount, Observe(Id), TimeStamp(Id))
  DefineProperty(XmlName, Output("Xml"), Filename("XML file (*.xml)"), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))
  DefineProperty(UseAppearanceMatrix, Observe(Id), TimeStamp(Id))
  DefineProperty(SsdRef, Observe(Id), TimeStamp(Id))
  DefineProperty(MiRef, Observe(Id), TimeStamp(Id))
  DefineProperty(SsdImg, Observe(Id), TimeStamp(Id))
  DefineProperty(MiImg, Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

AamAnalyser::AamAnalyser()
 : _From(-1), _To(1), _StepCount(10), _XmlName("out.xml"), _UseAppearanceMatrix(true),
   _SsdRef(true), _MiRef(true), _SsdImg(true), _MiImg(true), data(0)
{
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

TiXmlNode* AamAnalyser::createPlot(bool inReferenceFrame, SimilarityMeasure measure) const {
  AamMatchFunction objective(getImage(), getActiveAppearanceModel(), inReferenceFrame, measure, getUseAppearanceMatrix());

  const int count = getFocalShapeParameters()->size();
  vector<double> parameter(count);

  TiXmlNode* plot = new TiXmlElement("Plot");
  AamPlotResult result;

  for (int iMode = 0; iMode < count; ++iMode) {
    copy(getFocalShapeParameters()->begin(), getFocalShapeParameters()->end(), parameter.begin());
    TiXmlElement* modePlot = new TiXmlElement("ModePlot");
    if (getTargetShapeParameters() && (int)getTargetShapeParameters()->size() == count) {
      modePlot->SetAttribute("target", getTargetShapeParameters()->at(iMode) - getFocalShapeParameters()->at(iMode));
    }
    if (getStartShapeParameters() && (int)getStartShapeParameters()->size() == count) {
      modePlot->SetAttribute("start", getStartShapeParameters()->at(iMode) - getFocalShapeParameters()->at(iMode));
    }

    double from = getFrom() * getActiveAppearanceModel()->getSingularShapeParameters()->at(iMode);
    double to = getTo() * getActiveAppearanceModel()->getSingularShapeParameters()->at(iMode);
    for (int iStep = 0; iStep < getStepCount(); ++iStep) {
      double dStep = (1. - (double)iStep / (getStepCount()-1)) * from + (double)iStep / (getStepCount()-1) * to;

      //for (int i = 0; i < count; ++i)
      //  parameter[i] = dStep * getPlotDirection()[i] + getParameterVector()->at(i);
      parameter[iMode] = dStep + getFocalShapeParameters()->at(iMode);

      result.setStep(dStep);
      result.setX(parameter);
      result.setY(objective.eval(parameter));
      modePlot->LinkEndChild(capputils::Xmlizer::CreateXml(result));
    }
    plot->LinkEndChild(modePlot);
  }
  return plot;
}

TiXmlNode* AamAnalyser::createPlot3D(bool inReferenceFrame, SimilarityMeasure measure) const {
  AamMatchFunction objective(getImage(), getActiveAppearanceModel(), inReferenceFrame, measure, getUseAppearanceMatrix());

  const int count = getFocalShapeParameters()->size();
  vector<double> parameter(count);

  TiXmlNode* plot = new TiXmlElement("Plot");
  AamPlotResult result;

  TiXmlElement* plot3D = new TiXmlElement("Plot3D");
  plot3D->SetAttribute("rows", getStepCount());
  plot3D->SetAttribute("cols", getStepCount());

  if (getStartShapeParameters() && (int)getStartShapeParameters()->size() == count) {
    plot3D->SetAttribute("xstart", getStartShapeParameters()->at(0) - getFocalShapeParameters()->at(0));
    plot3D->SetAttribute("ystart", getStartShapeParameters()->at(1) - getFocalShapeParameters()->at(1));
  } else {
    if (!getStartShapeParameters())
      cout << "No shape parameters" << endl;
    if ((int)getStartShapeParameters()->size() == count) {
      cout << "Start count = " << getStartShapeParameters()->size() << ". Should be " << count << endl;
    }
  }
  if (getTargetShapeParameters() && (int)getTargetShapeParameters()->size() == count) {
    plot3D->SetAttribute("xtarget", getTargetShapeParameters()->at(0) - getFocalShapeParameters()->at(0));
    plot3D->SetAttribute("ytarget", getTargetShapeParameters()->at(1) - getFocalShapeParameters()->at(1));
  }

  double fromX = getFrom() * getActiveAppearanceModel()->getSingularShapeParameters()->at(0);
  double toX = getTo() * getActiveAppearanceModel()->getSingularShapeParameters()->at(0);
  double fromY = getFrom() * getActiveAppearanceModel()->getSingularShapeParameters()->at(1);
  double toY = getTo() * getActiveAppearanceModel()->getSingularShapeParameters()->at(1);

  copy(getFocalShapeParameters()->begin(), getFocalShapeParameters()->end(), parameter.begin());
  for (int y = 0; y < getStepCount(); ++y) {
    double dy = (1. - (double)y / (getStepCount()-1)) * fromY + (double)y / (getStepCount()-1) * toY;
    for (int x = 0; x < getStepCount(); ++x) {
      double dx = (1. - (double)x / (getStepCount()-1)) * fromX + (double)x / (getStepCount()-1) * toX;
        parameter[0] = dx + getFocalShapeParameters()->at(0);
        parameter[1] = dy + getFocalShapeParameters()->at(1);

        result.setXStep(dx);
        result.setYStep(dy);
        result.setX(parameter);
        result.setY(objective.eval(parameter));
        plot3D->LinkEndChild(capputils::Xmlizer::CreateXml(result));
    }
  }
  plot->LinkEndChild(plot3D);

  return plot;
}

void AamAnalyser::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamAnalyser();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getActiveAppearanceModel() || !getImage() || !getFocalShapeParameters())
  {
    return;
  }

  TiXmlNode* root = new TiXmlElement("Plots");
  TiXmlElement* plot;

  if (getSsdRef()) {
    plot = (TiXmlElement*)createPlot3D(true, SimilarityMeasure::SSD);
    plot->SetAttribute("label", "SSD in reference frame (3D)");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(100/8);
    plot = (TiXmlElement*)createPlot(true, SimilarityMeasure::SSD);
    plot->SetAttribute("label", "SSD in reference frame");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(200/8);
  }

  if (getMiRef()) {
    plot = (TiXmlElement*)createPlot3D(true, SimilarityMeasure::MI);
    plot->SetAttribute("label", "MI in reference frame (3D)");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(300/8);
    plot = (TiXmlElement*)createPlot(true, SimilarityMeasure::MI);
    plot->SetAttribute("label", "MI in reference frame");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(400/8);
  }

  if(getSsdImg()) {
    plot = (TiXmlElement*)createPlot3D(false, SimilarityMeasure::SSD);
    plot->SetAttribute("label", "SSD in image frame (3D)");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(500/8);
    plot = (TiXmlElement*)createPlot(false, SimilarityMeasure::SSD);
    plot->SetAttribute("label", "SSD in image frame");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(600/8);
  }

  if (getMiImg()) {
    plot = (TiXmlElement*)createPlot3D(false, SimilarityMeasure::MI);
    plot->SetAttribute("label", "MI in image frame (3D)");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(700/8);
    plot = (TiXmlElement*)createPlot(false, SimilarityMeasure::MI);
    plot->SetAttribute("label", "MI in image frame");
    root->LinkEndChild(plot);
    if (monitor)
      monitor->reportProgress(800/8);
  }

  capputils::Xmlizer::ToFile(getXmlName(), root);
}

void AamAnalyser::writeResults() {
  if (!data)
    return;
  setXmlName(getXmlName());
}

}

}
