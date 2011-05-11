#include "Paper.h"

#include <EventHandler.h>
#include <Verifier.h>
#include <ObserveAttribute.h>
#include <FileExists.h>
#include <FilenameAttribute.h>
#include <DescriptionAttribute.h>
#include <cmath>

#include "../gapputils/CsvReader.h"
#include "../gptest/gplib.h"
#include "../gptest/NLML.h"
#include "../gptest/NlmlCpu.h"

#include <Windows.h>

#ifdef RegisterClass
#undef RegisterClass
#endif

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

enum PropertyIds {
  AlgorithmId, TestId, TestfileId, IterationsCountId, SampleCountId, FeatureCountId,
  TrainTimeId, PredictionTimeId, FirstColumnId, LastColumnId, YColumnId, FirstRowId, LastRowId,
  RunId, ConfigurationId, ResultId
};

BeginPropertyDefinitions(Paper)

  ReflectableProperty(Algorithm, Observe(AlgorithmId))
  ReflectableProperty(Test, Observe(TestId))
  DefineProperty(Testfile, Observe(TestfileId), FileExists(), Filename())
  DefineProperty(IterationsCount, Observe(IterationsCountId))
  DefineProperty(SampleCount, Observe(SampleCountId))
  DefineProperty(FeatureCount, Observe(FeatureCountId))
  DefineProperty(TrainTime, Observe(TrainTimeId), Description("In ms"))
  DefineProperty(PredictionTime, Observe(PredictionTimeId), Description("In ms"))
  DefineProperty(FirstColumn, Observe(FirstColumnId))
  DefineProperty(LastColumn, Observe(LastColumnId))
  DefineProperty(YColumn, Observe(YColumnId))
  DefineProperty(FirstRow, Observe(FirstRowId))
  DefineProperty(LastRow, Observe(LastRowId))
  DefineProperty(Run, Observe(RunId))
  DefineProperty(ConfigurationName)
  DefineProperty(Result, Observe(ResultId))

EndPropertyDefinitions

Paper::Paper(void) : _Testfile(""), _IterationsCount(1), _SampleCount(0), _FeatureCount(0),
  _TrainTime(0), _PredictionTime(0), _FirstColumn(0), _LastColumn(1), _YColumn(2), _FirstRow(0),
  _LastRow(-1), _Run(true), _ConfigurationName("config.xml"), _Result(0)
{
  Changed.connect(capputils::EventHandler<Paper>(this, &Paper::changeEventHandler));
}


Paper::~Paper(void)
{
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

void Paper::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
  using namespace gplib;
  using namespace std;

  if (!capputils::Verifier::Valid(*this) || !getRun())
    return;

  if (eventId == TestfileId || eventId == IterationsCountId || eventId == AlgorithmId
      || eventId == TestId || (FirstColumnId <= eventId && eventId <= LastRowId) || eventId == RunId)
  {
    CsvReader xReader, yReader, xstarReader;
    xReader.setFirstColumn(getFirstColumn());
    xReader.setLastColumn(getLastColumn());
    xReader.setFirstRow(getFirstRow());
    xReader.setLastRow(getLastRow());
    xReader.setFilename(getTestfile());

    yReader.setFirstColumn(getYColumn());
    yReader.setLastColumn(getYColumn());
    yReader.setFirstRow(getFirstRow());
    yReader.setLastRow(getLastRow());
    yReader.setFilename(getTestfile());

    int d = xReader.getColumnCount();
    int n = xReader.getRowCount();

    //vector<float> x(xReader.getData(), xReader.getData() + d * n);
    double* data = xReader.getData();

    vector<float> x(d * n);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < n; ++j)
        x[i * n + j] = data[j * d + i];

    vector<float> y(yReader.getData(), yReader.getData() + n);
    vector<float> length(d);
    for (int i = 0; i < d; ++i)
      length[i] = 1.0f;

    vector<float> mean(d);
    vector<float> var(d);

    getMean(mean, x);
    getVar(var, x, mean);
    //standardize(x, mean, var);

    double result;

    switch (getTest()) {
    case Test::Training:
      {
        Algorithm algorithm = getAlgorithm();
        switch (algorithm) {
        case Algorithm::GpuNaive:
          {
            NLML nlml(&x[0], &y[0], n, d);
            int count = getIterationsCount();
            double time = GetTickCount();
            for (int i = 0; i < count; ++i)
              result = nlml.eval(1.f, 1.f, &length[0]);
            setTrainTime((double)(GetTickCount() - time) / (double)count);
          } break;

        case Algorithm::Cpu:
          {
            NlmlCpu nlml(&x[0], &y[0], n, d);
            int count = getIterationsCount();
            double time = GetTickCount();
            for (int i = 0; i < count; ++i)
              result = nlml.eval(1.f, 1.f, &length[0]);
            setTrainTime((double)(GetTickCount() - time) / (double)count);
          } break;
        }
        setResult(result);
      } break;

    case Test::Prediction:
      {
        GP& gp = GP::getInstance();
        vector<float> mu(n);
        vector<float> cov(n * n);
        int count = getIterationsCount();
        Algorithm algorithm = getAlgorithm();
        for (int i = 0; i < count; ++i) {
          switch(algorithm) {
          case Algorithm::GpuNaive:
            gp.gp(&mu[0], &cov[0], &x[0], &y[0], &x[0], n, d, n, 1.0f, 1.0f, &length[0]);
            break;

          case Algorithm::Cpu:
            gpcpu(&mu[0], &cov[0], &x[0], &y[0], &x[0], n, d, n, 1.0f, 1.0f, &length[0]);
            break;
          }
        }
      } break;
    }
    setSampleCount(n);
    setFeatureCount(d);
  }
}

}
