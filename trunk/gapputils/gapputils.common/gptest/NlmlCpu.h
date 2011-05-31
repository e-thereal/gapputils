#pragma once

#ifndef _GPLIB_NLMLCPU_H_
#define _GPLIB_NLMLCPU_H_

#include "gplib.h"

#include <vector>
#include <algorithm>

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
    std::copy(data, data + (rowCount * columnCount), this->data);
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

namespace gplib {

class GPTEST_API NlmlCpu
{
private:
  int n, d;
  std::vector<float> alpha, diag, x, y;
  Matrix<float> K;

public:
  NlmlCpu(float* x, float *y, int n, int d);
  ~NlmlCpu(void);

  double eval(float sigmaF, float sigmaN, float* length);
};

}

#endif