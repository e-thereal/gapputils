#include "GPRegression.h"

#include <algorithm>

#include <capputils/ObserveAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/Verifier.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/LabelAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <cmath>

#include "gpgpu.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace GaussianProcesses {

int GPRegression::xId;
int GPRegression::dId;

BeginPropertyDefinitions(GPRegression)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(FeatureCount, Observe(dId = PROPERTY_ID), Input(), ShortName("D"), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(X, Observe(xId = PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Y, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(TrainingCount, Observe(PROPERTY_ID), Input(), ShortName("N"), Volatile(), TimeStamp(PROPERTY_ID))

  DefineProperty(Xstar, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), ShortName("X*"), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Ystar, Observe(PROPERTY_ID), Output(), Hide(), ShortName("Y*"), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(TestCount, Observe(PROPERTY_ID), Input(), ShortName("M"), Volatile(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

GPRegression::GPRegression(void) : _FeatureCount(0), _X(0), _Y(0),
    _TrainingCount(0), _Xstar(0), _Ystar(0), _TestCount(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("GPRegression");
  Changed.connect(capputils::EventHandler<GPRegression>(this, &GPRegression::changeEventHandler));
}


GPRegression::~GPRegression(void)
{
  if (getYstar())
    delete getYstar();

  if (data)
    delete data;
}

void getMean(vector<float>& means, const vector<float>& values) {
  int rowCount = means.size();
  int columnCount = values.size() / rowCount;

  for (int row = 0, i = 0; row < rowCount; ++row) {
    float mean = 0;
    for (int col = 0; col < columnCount; ++col, ++i)
      mean += values[i];
    means[row] = mean / columnCount;
  }
}

void getVar(vector<float>& vars, const vector<float>& values, const vector<float>& means) {
  int rowCount = vars.size();
  int columnCount = values.size() / rowCount;

  for (int row = 0, i = 0; row < rowCount; ++row) {
    float var = 0;
    for (int col = 0; col < columnCount; ++col, ++i)
      var += (means[row] - values[i]) * (means[row] - values[i]);
    vars[row] = var / columnCount;
  }
}

void standardize(vector<float>& values, const vector<float>& means, const vector<float>& vars) {
  int rowCount = vars.size();
  int columnCount = values.size() / rowCount;
  for (int row = 0, i = 0; row < rowCount; ++row) {
    float invstdd = vars[row] > 1 ? 1.f / sqrt(vars[row]) : 1.0f;
    for (int col = 0; col < columnCount; ++col, ++i)
      values[i] = (values[i] - means[row]) * invstdd;
  }
}

void GPRegression::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
}

void GPRegression::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  float sigmaF = 1.f, sigmaN = 1.f;

  if (!data)
    data = new GPRegression();

  if (!capputils::Verifier::Valid(*this))
    return;

  // Train the GP

  int n = getTrainingCount();
  int d = getFeatureCount();
  int m = getTestCount();

  vector<float> x(n * d);
  vector<float> y(n);
  vector<float> xstar(m * d);
  vector<float> ystar(m);
  vector<float> length(d);

  copy(getX(), getX() + (d*n), x.begin());
  copy(getY(), getY() + n, y.begin());
  copy(getXstar(), getXstar() + (d*m), xstar.begin());

  vector<float> means(d);
  vector<float> vars(d);
  getMean(means, x);
  getVar(vars, x, means);
  standardize(x, means, vars);
  standardize(xstar, means, vars);

  trainGP(sigmaF, sigmaN, &length[0], &x[0], &y[0], n, d, monitor);

  // Make predictions with the GP
  gp(&ystar[0], 0, &x[0], &y[0], &xstar[0], n, d, m, sigmaF, sigmaN, &length[0]);

  if (data->getYstar())
    delete data->getYstar();

  double* result = new double[m];
  copy(ystar.begin(), ystar.end(), result);
  data->setYstar(result);
}

void GPRegression::writeResults() {
  if (!data)
    return;

  if (getYstar())
    delete getYstar();

  setYstar(data->getYstar());
  data->setYstar(0);
}

}
