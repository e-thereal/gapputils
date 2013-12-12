/*
 * ConvolutionType.h
 *
 *  Created on: Dec 9, 2013
 *      Author: tombr
 */

#ifndef GML_CONVOLUTIONTYPE_H_
#define GML_CONVOLUTIONTYPE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(ConvolutionType, Circular, Valid);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm4d::ConvolutionType);

#endif /* GML_CONVOLUTIONTYPE_H_ */
