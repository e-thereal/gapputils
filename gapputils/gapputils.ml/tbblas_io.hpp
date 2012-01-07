#pragma once
#ifndef GAPPUTILS_TBBLAS_IO_
#define GAPPUTILS_TBBLAS_IO_

#include <tbblas/device_matrix.hpp>
#include <thrust/copy.h>

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

#include "ublas_io.hpp"

/**
 * \brief Column-major storage 
 */
template<class T>
bool read_matrix(FILE* file, tbblas::device_matrix<T>& m) {
  unsigned magic, rowCount, columnCount;
  if (fread(&magic, sizeof(unsigned), 1, file) != 1)
    return false;
  if (!check_magic<T>(magic))
    return false;

  if (fread(&rowCount, sizeof(unsigned), 1, file) != 1)
    return false;
  if (fread(&columnCount, sizeof(unsigned), 1, file) != 1)
    return false;

  const unsigned count = rowCount * columnCount;
  if (rowCount != m.size1() || columnCount != m.size2())
    m.resize(rowCount, columnCount);
  T* buffer = new T[count];
  if (fread(buffer, sizeof(T), count, file) != count)
    return false;

  thrust::copy(buffer, buffer + count, m.data().begin());

  return true;
}

template<class T>
bool read_vector(FILE* file, tbblas::device_vector<T>& v) {
  unsigned magic, rowCount, columnCount;
  if (fread(&magic, sizeof(unsigned), 1, file) != 1)
    return false;
  if (!check_magic<T>(magic))
    return false;

  if (fread(&rowCount, sizeof(unsigned), 1, file) != 1)
    return false;
  if (fread(&columnCount, sizeof(unsigned), 1, file) != 1)
    return false;

  const unsigned count = rowCount * columnCount;
  if (count != v.size())
    v.resize(count);
  T* buffer = new T[count];
  if (fread(buffer, sizeof(T), count, file) != count)
    return false;

  thrust::copy(buffer, buffer + count, v.data().begin());

  return true;
}

template<class T>
bool read_matrix(const std::string& filename, tbblas::device_matrix<T>& m) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file)
    return false;
  bool res = read_matrix(file, m);
  fclose(file);

  return res;
}

template<class T>
bool read_vector(const std::string& filename, tbblas::device_vector<T>& v) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file)
    return false;
  bool res = read_vector(file, v);
  fclose(file);

  return res;
}

template<class T>
bool read_matrix_from_text(const std::string& filename, tbblas::device_matrix<T>& m) {
  std::ifstream file(filename.c_str());
  if (!file)
    return false;

  std::string line;
  int rowCount = 0, columnCount = 0;
  std::vector<T> values;
  T value;
  while(getline(file, line)) {
    ++columnCount;
    std::stringstream lineStream(line);
    int rows = 0;
    while(!lineStream.eof()) {
      ++rows;
      lineStream >> value;
      values.push_back(value);
    }
    rowCount = std::max(rowCount, rows);
  }
  file.close();

  if (m.rowCount() != rowCount || m.columnCount() != columnCount)
    return false;

  thrust::copy(values.begin(), values.end(), m.data().begin());

  return true;
}

template<class T>
bool read_vector_from_text(const std::string& filename, tbblas::device_vector<T>& v) {
  std::ifstream file(filename.c_str());
  if (!file)
    return false;

  int count = 0;
  std::vector<T> values;
  T value;
  while(!file.eof()) {
    ++count;
    file >> value;
    values.push_back(value);
  }
  file.close();

  if (v.size() != count)
    return false;

  thrust::copy(values.begin(), values.end(), v.data().begin());

  return true;
}

#endif /* GAPPUTILS_TBBLAS_IO_ */
