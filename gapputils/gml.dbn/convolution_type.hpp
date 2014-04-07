/*
 * ConvolutionType.h
 *
 *  Created on: Dec 9, 2013
 *      Author: tombr
 */

#ifndef GML_DBN_CONVOLUTIONTYPE_H_
#define GML_DBN_CONVOLUTIONTYPE_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace dbn {

CapputilsEnumerator(convolution_type, Circular, Valid);

}

}

DefineEnumeratorSerializeTrait(gml::dbn::convolution_type);

#endif /* GML_DBN_CONVOLUTIONTYPE_H_ */
