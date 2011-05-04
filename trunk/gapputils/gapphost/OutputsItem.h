/*
 * OutputsItem.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef _GAPPUTILS_OUTPUTSITEM_H_
#define _GAPPUTILS_OUTPUTSITEM_H_

#include "ToolItem.h"

namespace gapputils {

class OutputsItem : public ToolItem {
public:
  OutputsItem(workflow::Node* node, Workbench *bench = 0);
  virtual ~OutputsItem();

  virtual std::string getLabel() const;
  virtual void updateConnections();
};

}

#endif /* OUTPUTSITEM_H_ */
