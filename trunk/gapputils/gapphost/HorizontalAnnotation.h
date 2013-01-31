/*
 * HorizontalAnnotation.h
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#ifndef GML_HORIZONTALANNOTATION_H_
#define GML_HORIZONTALANNOTATION_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace interfaces {

class HorizontalAnnotation : public DefaultWorkflowElement<HorizontalAnnotation> {

  InitReflectableClass(HorizontalAnnotation)

public:
  HorizontalAnnotation();
};

} /* namespace interfaces */

#endif /* HORIZONTALANNOTATION_H_ */
