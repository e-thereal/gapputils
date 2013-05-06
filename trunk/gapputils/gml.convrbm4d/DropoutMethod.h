/*
 * DropoutMethod.h
 *
 *  Created on: Apr 19, 2013
 *      Author: tombr
 */

#ifndef GML_DROPOUTMETHOD_H_
#define GML_DROPOUTMETHOD_H_

#include <capputils/Enumerators.h>

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(DropoutStage, Epoch, Batch, Sample);
CapputilsEnumerator(DropoutMethod, DropColumn, DropIndividual);

}

}

DefineEnumeratorSerializeTrait(gml::convrbm4d::DropoutStage);
DefineEnumeratorSerializeTrait(gml::convrbm4d::DropoutMethod);

#endif /* GML_DROPOUTMETHOD_H_ */
