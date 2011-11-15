#include "ublas_io.hpp"

template<>
bool check_magic<float>(unsigned magic) {
  return magic == 0x1;
}

template<>
bool check_magic<double>(unsigned magic) {
  return magic == 0x2;
}
