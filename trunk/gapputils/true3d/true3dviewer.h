#ifndef TRUE3DVIEWER_H
#define TRUE3DVIEWER_H

#include "true3d_global.h"

#include <gapputils/WorkflowElement.h>

namespace true3d {

class True3dViewer : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(True3dViewer)

private:
  mutable True3dViewer* data;

public:
  True3dViewer();
  virtual ~True3dViewer();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);

  virtual void show();
};

}

#endif // TRUE3DVIEWER_H
