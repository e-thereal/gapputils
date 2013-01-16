/*
 * CodingDirection.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_CODINGDIRECTION_H_
#define GML_CODINGDIRECTION_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace rbm {

CapputilsEnumerator(CodingDirection, Encode, Decode);

}

}

DefineEnumeratorSerializeTrait(gml::rbm::CodingDirection);

#endif /* GML_CODINGDIRECTION_H_ */
