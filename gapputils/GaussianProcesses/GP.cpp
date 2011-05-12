#include "GP.h"

#include <ObserveAttribute.h>
#include <EventHandler.h>
#include <OutputAttribute.h>
#include <FilenameAttribute.h>
#include <NotEqualAssertion.h>
#include <Verifier.h>
#include <Xmlizer.h>
#include <EnumerableAttribute.h>
#include <LabelAttribute.h>
#include <ShortNameAttribute.h>
#include <VolatileAttribute.h>
#include <cmath>

//#include "../gptest/gplib.h"
//#include "../gptest/NLML.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace GaussianProcesses {

enum PropertyIds {
  LabelId, XId, YId, OutputId, FirstId, StepId, LastId, XstarId, MuId, CIId, SigmaId,
  LengthId, Sigma2Id, AutoId, TrainId
};

BeginPropertyDefinitions(GP)

DefineProperty(Label, Observe(LabelId), Label())
DefineProperty(X, Observe(XId), Enumerable<vector<float>, false>())
DefineProperty(Y, Observe(YId), Enumerable<vector<float>, false>())
DefineProperty(OutputName, Observe(OutputId), ShortName("Xml"), Output(), Filename())
DefineProperty(First, Observe(FirstId))
DefineProperty(Step, Observe(StepId))
DefineProperty(Last, Observe(LastId))
DefineProperty(Xstar, Observe(XstarId), Enumerable<vector<float>, false>())
DefineProperty(Mu, Observe(MuId), Enumerable<vector<float>, false>())
DefineProperty(CI, Observe(CIId), Enumerable<vector<float>, false>())
DefineProperty(SigmaF, Observe(SigmaId))
DefineProperty(Length, Observe(LengthId))
DefineProperty(SigmaN, Observe(Sigma2Id))
DefineProperty(Auto, Observe(AutoId), Volatile())
DefineProperty(Train, Observe(TrainId), Volatile())

EndPropertyDefinitions

GP::GP(void) : _Label("GP"), _OutputName(""), _First(0.0), _Step(0.25), _Last(0.0), _SigmaF(1.0),
               _Length(1.0), _SigmaN(1.0), _Auto(false), _Train(false)
{
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
  Changed.connect(capputils::EventHandler<GP>(this, &GP::changeHandler));
}


GP::~GP(void)
{
}

void GP::changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (getX().size() != getY().size())
    return;

  if (eventId == FirstId || eventId == StepId || eventId == LastId) {
    // Recompute xstar
    vector<float> xstar;
    for (float x = getFirst(); x <= getLast(); x += getStep())
      xstar.push_back(x);
    setXstar(xstar);
  }

  if (getAuto() && (eventId == XId || eventId == YId || eventId == XstarId ||
      eventId == SigmaId || eventId == LengthId || eventId == Sigma2Id || eventId == AutoId)) {
    // recompute Mu
    vector<float> x = getX();
    vector<float> y = getY();
    vector<float> xstar = getXstar();
    vector<float> mu(xstar.size());
    vector<float> ci(xstar.size());

    float *cov = new float[xstar.size() * xstar.size()];

    //gplib::GP& gp = gplib::GP::getInstance();

    float length = getLength();

    //gp.gp(&mu[0], cov, &x[0], &y[0], &xstar[0], x.size(), 1, xstar.size(), getSigmaF(),
    //   getSigmaN(), &length);

    for (int i = 0; i < xstar.size(); ++i)
      ci[i] = 2 * sqrt(cov[i + i * xstar.size()]);

    //gplib::NLML nlml(&x[0], &y[0], x.size(), 1);
    //cout << "nlml = " << nlml.eval(getSigmaF(), getSigmaN(), &length) << endl;

    setCI(ci);
    setMu(mu);

    delete cov;
  }

  if (eventId == TrainId && getTrain()) {
    bool automatic = getAuto();
    setAuto(false);
    //gplib::GP& gp = gplib::GP::getInstance();
    float sigmaF = getSigmaF();
    float sigmaN = getSigmaN();
    float length = getLength();

    vector<float> x = getX();
    vector<float> y = getY();

    //gp.trainGP(sigmaF, sigmaN, &length, &x[0], &y[0], x.size(), 1);

    setSigmaF(sigmaF);
    setSigmaN(sigmaN);
    setLength(length);

    setAuto(automatic);
  }

  if ((eventId == MuId) && capputils::Verifier::Valid(*this) && getAuto()) {
    capputils::Xmlizer::ToXml("gp.xml", *this);
    setOutputName("gp.xml");
  }
}

}
