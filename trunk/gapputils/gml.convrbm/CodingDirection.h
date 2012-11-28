/*
 * CodingDirection.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef CODINGDIRECTION_H_
#define CODINGDIRECTION_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm {

CapputilsEnumerator(CodingDirection, Encode, Decode);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm::CodingDirection);

#endif /* CODINGDIRECTION_H_ */
