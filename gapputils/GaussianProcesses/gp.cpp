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

#include <cublas.h>

#include "NLML.h"

using namespace gapputils::workflow;
using namespace optlib;
using namespace std;

namespace GaussianProcesses {

/// Row major matrix (as typical C style)
template<class T>
class Matrix {
private:
  T* data;
  int rowCount, columnCount;

public:
  Matrix(int rowCount, int columnCount) : rowCount(rowCount), columnCount(columnCount) {
    data = new T[rowCount * columnCount];
  }

  Matrix (T data[], int rowCount, int columnCount) : rowCount(rowCount), columnCount(columnCount) {
    this->data = new T[rowCount * columnCount];
    copy(data, data + (rowCount * columnCount), this->data);
  }

  virtual ~Matrix() {
    delete data;
  }

  T* getData() const {
    return data;
  }

  int getRowCount() const {
    return rowCount;
  }

  int getColumnCount() const {
    return columnCount;
  }

  bool isQuadratic() const {
    return getRowCount() == getColumnCount();
  }

  T& getElement(int rowIndex, int columnIndex) {
    return data[rowIndex * getColumnCount() + columnIndex];
  }

  T& operator()(int rowIndex, int columnIndex) {
    return getElement(rowIndex, columnIndex);
  }
};

template<class T>
ostream& operator<<(ostream& stream, const Matrix<T>& matrix) {
  const int rows = matrix.getRowCount();
  const int cols = matrix.getColumnCount();
  const T* data = matrix.getData();

  stream << "[ ";
  for (int i = 0, y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x, ++i) {
      stream << data[i];
      if (x < cols - 1)
        stream << " ";
    }
    if (y < rows - 1)
      stream << "; ";
  }
  stream << " ]";

  return stream;
}

void printMatrix(ostream& stream, const char* name, float *d_m, int m, int n, int pitch) {
  Matrix<float> M(n, m);

  //cublasGetVector(m * n, sizeof(float), d_m, 1, M.getData(), 1);
  cublasGetMatrix(m, n, sizeof(float), d_m, pitch, M.getData(), m);
  stream << name << " = " << M << "';" << endl;
}


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
  for (unsigned i = 0; i < params.size(); ++i)
    params[i] = 0;

  GridParameter gridParams;

  // Sigma f
  vector<double> sigmaFDir(3);
  sigmaFDir[0] = 1.0;
  sigmaFDir[1] = 0.0;
  sigmaFDir[2] = 0.0;

  GridLineParameter sigmaFLine;
  sigmaFLine.direction = sigmaFDir;
  sigmaFLine.minValue = -3;
  sigmaFLine.maxValue = 3;
  sigmaFLine.samplesCount = 3;
  gridParams.gridLines.push_back(sigmaFLine);

  // Sigma n
  vector<double> sigmaNDir(3);
  sigmaNDir[0] = 0.0;
  sigmaNDir[1] = 1.0;
  sigmaNDir[2] = 0.0;

  GridLineParameter sigmaNLine;
  sigmaNLine.direction = sigmaNDir;
  sigmaNLine.minValue = -3;
  sigmaNLine.maxValue = 3;
  sigmaNLine.samplesCount = 3;
  gridParams.gridLines.push_back(sigmaNLine);

  // Length
  vector<double> lengthDir(2 + d);
  lengthDir[0] = 0.0;
  lengthDir[1] = 0.0;
  for (int i = 0; i < d; ++i)
    lengthDir[i + 2] = 1.0;

  GridLineParameter lengthLine;
  lengthLine.direction = lengthDir;
  lengthLine.minValue = -3;
  lengthLine.maxValue = 3;
  lengthLine.samplesCount = 3;
  gridParams.gridLines.push_back(lengthLine);

  GridSamplingOptimizer<ConjugateGradientsOptimizer> optimizer;
  optimizer.setParameter(GridSampling, &gridParams);
  //ConjugateGradientsOptimizer optimizer;
  optimizer.setGradientMethod(SteepestDescentOptimizer::Analytic);
  //DownhillSimplexOptimizer optimizer;

  ProgressListener listener(monitor, sigmaFLine.samplesCount * sigmaNLine.samplesCount * lengthLine.samplesCount);
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
