#ifndef GAPPUTILS_H
#define GAPPUTILS_H

#ifdef _WIN32
#ifndef GAPPUTILS_EXPORTS
#pragma comment (lib, "gapputils")
#endif
#endif

namespace gapputils {

void registerClasses();

}

#endif // GAPPUTILS_H
