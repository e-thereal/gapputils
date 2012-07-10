#ifndef GAPPUTILS_H
#define GAPPUTILS_H

#ifdef _WIN32
#ifndef GAPPUTILS_EXPORTS
#pragma comment (lib, "gapputils")
#endif
#endif

#include <boost/crc.hpp>

namespace gapputils {
  typedef boost::crc_32_type::value_type checksum_t;
}

#endif // GAPPUTILS_H
