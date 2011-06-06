/*
 * gp.cpp
 *
 *  Created on: Jun 3, 2011
 *      Author: tombr
 */

#include "gpgpu.h"

#include <cmath>

#include <optlib/ConjugateGradientsOptimizer.h>
#include <optlib/DownhillSimplexOptimizer.h>
#include <optlib/OptimizerException.h>
#include <optlib/GridSamplingOptimizer.h>

#include "NLML.h"

using namespace gapputils::workflow;
using namespace optlib;
using namespace std;

namespace GaussianProcesses {

class ProgressListener : public optlib::ConjugateGradientsOptimizer::ObserverType {
private:
  IProgressMonitor* monitor;
  int i, count;

public:
  ProgressListener(IProgressMonitor* monitor, int count) : monitor(monitor), i(0), count(count) { }

   virtual void eventTriggered(const IOptimizerEvent& event, IOptimizer<ConjugateGradientsOptimizer::DomainType>& sender) {
     const DirectionSetNewSolution* newSolution = dynamic_cast<const DirectionSetNewSolution*>(&event);
     if (newSolution) {
       if (newSolution->phase == ProgressEvent::Start) {
         if (monitor)
          monitor->reportProgress(++i * 100 / count);
       }
     }
   }
};

void trainGP(float& sigmaF, float& sigmaN, float* length, float* x, float* y, int n, int d, IProgressMonitor* monitor) {
  NLML nlml(x, y, n, d);

  vector<double> params(2 + d);
  for (int i = 0; i < params.size(); ++i)
    params[i] = 0;

  GridParameter gridParams;

  // Sigma f
  vector<double> sigmaFDir(3);
  sigmaFDir[0] = 1.0;
  sigmaFDir[1] = 0.0;
  sigmaFDir[2] = 0.0;

  GridLineParameter sigmaFLine;
  sigmaFLine.direction = sigmaFDir;
  sigmaFLine.minValue = -2;
  sigmaFLine.maxValue = 2;
  sigmaFLine.samplesCount = 3;
  gridParams.gridLines.push_back(sigmaFLine);

  // Sigma n
  vector<double> sigmaNDir(3);
  sigmaNDir[0] = 0.0;
  sigmaNDir[1] = 1.0;
  sigmaNDir[2] = 0.0;

  GridLineParameter sigmaNLine;
  sigmaNLine.direction = sigmaNDir;
  sigmaNLine.minValue = -2;
  sigmaNLine.maxValue = 2;
  sigmaNLine.samplesCount = 5;
  gridParams.gridLines.push_back(sigmaNLine);

  // Length
  vector<double> lengthDir(2 + d);
  lengthDir[0] = 0.0;
  lengthDir[1] = 0.0;
  for (int i = 0; i < d; ++i)
    lengthDir[i + 2] = 1.0;

  GridLineParameter lengthLine;
  lengthLine.direction = lengthDir;
  lengthLine.minValue = -2;
  lengthLine.maxValue = 2;
  lengthLine.samplesCount = 5;
  gridParams.gridLines.push_back(lengthLine);

  GridSamplingOptimizer<ConjugateGradientsOptimizer> optimizer;
  optimizer.setParameter(GridSampling, &gridParams);
  //DownhillSimplexOptimizer optimizer;

  ProgressListener listener(monitor, 3 * 5 * 5);
  optimizer.addObserver(listener);

  try {
    optimizer.minimize(params, nlml);
  } catch (optlib::OptimizerException ex) {
    cout << ex.what() << endl;
  }

  sigmaF = exp(params[0]);
  sigmaN = exp(params[1]);
  for (int i = 0; i < d; ++i)
    length[i] = exp(params[i + 2]);
}

}
