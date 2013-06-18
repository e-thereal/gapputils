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

namespace dimreduce {

CapputilsEnumerator(CodingDirection, Encode, Decode);

}

}

DefineEnumeratorSerializeTrait(gml::dimreduce::CodingDirection);

#endif /* CODINGDIRECTION_H_ */
