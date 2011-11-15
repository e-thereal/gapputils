#pragma once
#ifndef GAPPUTILS_TBBLAS_IO_
#define GAPPUTILS_TBBLAS_IO_

#include <tbblas/device_matrix.hpp>
#include <thrust/copy.h>

#include <cstdio>
#include <algorithm>

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

#endif /* GAPPUTILS_TBBLAS_IO_ */
