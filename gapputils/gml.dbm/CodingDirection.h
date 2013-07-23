/*
 * CodingDirection.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_DBM_CODINGDIRECTION_H_
#define GML_DBM_CODINGDIRECTION_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace dbm {

CapputilsEnumerator(CodingDirection, Encode, Decode);

}

}

DefineEnumeratorSerializeTrait(gml::dbm::CodingDirection);

#endif /* GML_DBM_CODINGDIRECTION_H_ */
