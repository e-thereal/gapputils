/*
 * Interfaces.h
 *
 *  Created on: 2012-10-21
 *      Author: tombr
 */

#ifndef INTERFACES_H_
#define INTERFACES_H_

#include <capputils/ReflectableClass.h>

namespace gapputils {

class Interfaces : public capputils::reflection::ReflectableClass {
  InitReflectableClass(Interfaces)

  Property(Integer, int)
};

} /* namespace gapputils */
#endif /* INTERFACES_H_ */
