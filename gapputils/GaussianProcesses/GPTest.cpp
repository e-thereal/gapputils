#ifdef DEBUG
#undef DEBUG
#endif
#define DEBUG 1

#include "GPTest.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/OutputAttribute.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/Verifier.h>
#include <capputils/Xmlizer.h>
#include <capputils/EnumerableAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <cmath>

#include "gpgpu.h"
#include "NLML.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace GaussianProcesses {

enum PropertyIds {
  LabelId, XId, YId, OutputId, FirstId, StepId, LastId, XstarId, MuId, CIId, SigmaId,
  LengthId, Sigma2Id, AutoId, TrainId
};

BeginPropertyDefinitions(GPTest)

DefineProperty(Label, Observe(LabelId), Label())
DefineProperty(X, Observe(XId), Enumerable<vector<float>, false>(), TimeStamp(PROPERTY_ID))
DefineProperty(Y, Observe(YId), Enumerable<vector<float>, false>(), TimeStamp(PROPERTY_ID))
DefineProperty(OutputName, Observe(OutputId), ShortName("Xml"), Output(), Filename(), TimeStamp(PROPERTY_ID))
DefineProperty(First, Observe(FirstId), TimeStamp(PROPERTY_ID))
DefineProperty(Step, Observe(StepId), TimeStamp(PROPERTY_ID))
DefineProperty(Last, Observe(LastId), TimeStamp(PROPERTY_ID))
DefineProperty(Xstar, Observe(XstarId), Enumerable<vector<float>, false>(), TimeStamp(PROPERTY_ID))
DefineProperty(Mu, Observe(MuId), Enumerable<vector<float>, false>(), TimeStamp(PROPERTY_ID))
DefineProperty(CI, Observe(CIId), Enumerable<vector<float>, false>(), TimeStamp(PROPERTY_ID))
DefineProperty(SigmaF, Observe(SigmaId), TimeStamp(PROPERTY_ID))
DefineProperty(Length, Observe(LengthId), TimeStamp(PROPERTY_ID))
DefineProperty(SigmaN, Observe(Sigma2Id), TimeStamp(PROPERTY_ID))
DefineProperty(Train, Observe(TrainId), Volatile(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

GPTest::GPTest(void) : _Label("GPTest"), _OutputName("gp.xml"), _First(0.0), _Step(0.25), _Last(0.0), _SigmaF(1.0),
               _Length(1.0), _SigmaN(1.0), _Train(false), data(0)
{
  WfeUpdateTimestamp

  _X.push_back(1);
  _X.push_back(2);
  _X.push_back(4);
  _X.push_back(5);
  _X.push_back(6);
  _X.push_back(9);

  _Y.push_back(5);
  _Y.push_back(3);
  _Y.push_back(1);
  _Y.push_back(1);
  _Y.push_back(2);
  _Y.push_back(4);
  Changed.connect(capputils::EventHandler<GPTest>(this, &GPTest::changeHandler));
}

// Segmentation fault here when module is updated.
GPTest::~GPTest(void)
{
  if (data)
    delete data;
}

void GPTest::changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (getX().size() != getY().size())
    return;

  if (eventId == FirstId || eventId == StepId || eventId == LastId) {
    // Recompute xstar
    vector<float> xstar;
    for (float x = getFirst(); x <= getLast(); x += getStep())
      xstar.push_back(x);
    setXstar(xstar);
  }
}

void GPTest::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new GPTest();

  float sigmaF = getSigmaF();
  float sigmaN = getSigmaN();
  float length = getLength();

  if (getTrain()) {
    vector<float> x = getX();
    vector<float> y = getY();

    trainGP(sigmaF, sigmaN, &length, &x[0], &y[0], x.size(), 1, monitor);
    data->setSigmaF(sigmaF);
    data->setSigmaN(sigmaN);
    data->setLength(length);
  }

  vector<float> x = getX();
  vector<float> y = getY();
  vector<float> xstar = getXstar();
  vector<float> mu(xstar.size());
  vector<float> ci(xstar.size());

  float *cov = new float[xstar.size() * xstar.size()];

  gp(&mu[0], cov, &x[0], &y[0], &xstar[0], x.size(), 1, xstar.size(), sigmaF,
      sigmaN, &length);

  for (int i = 0; i < xstar.size(); ++i)
    ci[i] = 2 * sqrt(cov[i + i * xstar.size()]);

  NLML nlml(&x[0], &y[0], x.size(), 1);
  cout << "nlml = " << nlml.eval(sigmaF, sigmaN, &length) << endl;
  nlml.gradient(sigmaF, sigmaN, &length);

  data->setCI(ci);
  data->setMu(mu);

  delete cov;
}

void GPTest::writeResults() {
  if (!data)
    return;

  if (getTrain()) {
    setSigmaF(data->getSigmaF());
    setSigmaN(data->getSigmaN());
    setLength(data->getLength());
  }
  setCI(data->getCI());
  setMu(data->getMu());

  capputils::Xmlizer::ToXml(getOutputName(), *this);
  setOutputName(getOutputName());
}

}
