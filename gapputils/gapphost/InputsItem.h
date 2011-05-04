/*
 * InputsItem.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef _GAPPUTILS_INPUTSITEM_H_
#define _GAPPUTILS_INPUTSITEM_H_

#include "ToolItem.h"

namespace gapputils {

class InputsItem : public ToolItem {
public:
  InputsItem(workflow::Node* node, Workbench *bench = 0);
  virtual ~InputsItem();

  virtual std::string getLabel() const;
  virtual void updateConnections();
};

}

#endif /* INPUTSITEM_H_ */
