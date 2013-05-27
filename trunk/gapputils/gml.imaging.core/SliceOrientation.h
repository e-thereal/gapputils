/*
 * SliceOrientation.h
 *
 *  Created on: 2013-05-24
 *      Author: tombr
 */

#ifndef GML_SLICEORIENTATION_H_
#define GML_SLICEORIENTATION_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(SliceOrientation, Axial, Sagital, Coronal);

}

}

}

DefineEnumeratorSerializeTrait(gml::imaging::core::SliceOrientation);

#endif /* GML_SLICEORIENTATION_H_ */
