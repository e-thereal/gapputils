#include "AamTester.h"

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

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <culib/lintrans.h>

#include <algorithm>

#include "AamGenerator.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(AamTester)
  ReflectableBase(gapputils::workflow::WorkflowElement)
  
  DefineProperty(ActiveAppearanceModel, Input("AAM"), Hide(), Volatile(), Reflectable<boost::shared_ptr<ActiveAppearanceModel> >(), Observe(Id), TimeStamp(Id))
  DefineProperty(SampleImage, Output("Img"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(SampleGrid, Output("Grid"), Hide(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(FirstMode, Observe(Id), TimeStamp(Id))
  DefineProperty(SecondMode, Observe(Id), TimeStamp(Id))
EndPropertyDefinitions

AamTester::AamTester(void) : _FirstMode(0), _SecondMode(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("AamTester");

  Changed.connect(capputils::EventHandler<AamTester>(this, &AamTester::changedHandler));
}

AamTester::~AamTester(void)
{
  if (data)
    delete data;
}

void AamTester::changedHandler(capputils::ObservableClass*, int) {

}

void AamTester::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new AamTester();

  if (!getActiveAppearanceModel())
    return;

  boost::shared_ptr<ActiveAppearanceModel> model = getActiveAppearanceModel();
  boost::shared_ptr<vector<float> > sap = model->getSingularAppearanceParameters();

  const int mpCount = model->getAppearanceParameterCount();

  boost::shared_ptr<vector<float> >parameters(new vector<float>(mpCount));
  for (int i = 0; i < mpCount; ++i)
    (*parameters)[i] = 0;
  (*parameters)[0] = getFirstMode() * sap->at(0);
  (*parameters)[1] = getSecondMode() * sap->at(1);

  AamGenerator gen;
  gen.setActiveAppearanceModel(model);
  gen.setParameterVector(parameters);
  gen.execute(0);
  gen.writeResults();

  data->setSampleImage(gen.getOutputImage());
  data->setSampleGrid(gen.getOutputGrid());
}

void AamTester::writeResults() {
  if (!data)
    return;

  setSampleImage(data->getSampleImage());
  setSampleGrid(data->getSampleGrid());
}

}

}
