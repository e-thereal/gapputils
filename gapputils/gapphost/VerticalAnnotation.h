/*
 * VerticalAnnotation.h
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#ifndef GML_VERTICALANNOTATION_H_
#define GML_VERTICALANNOTATION_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace interfaces {

class VerticalAnnotation : public DefaultWorkflowElement<VerticalAnnotation> {

  InitReflectableClass(VerticalAnnotation)

public:
  VerticalAnnotation();
};

} /* namespace interfaces */

#endif /* GML_VERTICALANNOTATION_H_ */
