#pragma once
#ifndef GAPPUTILS_UBLAS_IO_
#define GAPPUTILS_UBLAS_IO_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>

#include <cstdio>
#include <algorithm>

template<class T>
bool check_magic(unsigned magic) {
  return false;
}

template<>
bool check_magic<float>(unsigned magic) {
  return magic == 0x1;
}

template<>
bool check_magic<double>(unsigned magic) {
  return magic == 0x2;
}

/**
 * \brief Column-major storage 
 */
template<class T>
bool read_matrix(FILE* file, boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major>& m) {
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
    m.resize(rowCount, columnCount, false);
  T* buffer = new T[count];
  if (fread(buffer, sizeof(T), count, file) != count)
    return false;

  std::copy(buffer, buffer + count, m.data().begin());

  return true;
}

template<class T>
bool read_vector(FILE* file, boost::numeric::ublas::vector<T>& v) {
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
    v.resize(count, false);
  T* buffer = new T[count];
  if (fread(buffer, sizeof(T), count, file) != count)
    return false;

  std::copy(buffer, buffer + count, v.begin());

  return true;
}

template<class T>
bool read_matrix(const std::string& filename, boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major>& m) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file)
    return false;
  bool res = read_matrix(file, m);
  fclose(file);

  return res;
}

template<class T>
bool read_vector(const std::string& filename, boost::numeric::ublas::vector<T>& v) {
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file)
    return false;
  bool res = read_vector(file, v);
  fclose(file);

  return res;
}

#endif