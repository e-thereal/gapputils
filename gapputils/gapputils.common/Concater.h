/*
 * Concat.h
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#ifndef _GAPPUTILS_CONCAT_H_
#define _GAPPUTILS_CONCAT_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {

namespace common {

class Concater : public workflow::DefaultWorkflowElement {

  InitReflectableClass(Concater)

  Property(Input1, std::string)
  Property(Input2, std::string)
  Property(Output, std::string)
  Property(Separator, std::string)

private:
  static int outputId;

public:
  Concater();
  virtual ~Concater();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* CONCAT_H_ */
