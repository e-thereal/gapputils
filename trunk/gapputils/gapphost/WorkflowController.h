#pragma once
#ifndef _GAPPUTILS_WORKFLOWCONTROLLER_H_
#define _GAPPUTILS_WORKFLOWCONTROLLER_H_

#include "Workbench.h"
#include "Graph.h"

#include <qobject.h>

namespace gapputils {

namespace workflow {

class Controller : public QObject
{
  Q_OBJECT

private:
  static Controller* instance;
  Workbench* workbench;

protected:
  Controller(void);

public:
  virtual ~Controller(void);

  static Controller& getInstance();
  void setWorkbench(Workbench* workbench);

  void newModule(const std::string& name);
  void newItem(Node* node);
  void resumeFromModel();

private Q_SLOTS:
  void itemChangedHandler(ToolItem* item);
  void deleteItem(ToolItem* item);
};

}

}

#endif
