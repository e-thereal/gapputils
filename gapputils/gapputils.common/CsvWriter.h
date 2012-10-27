#pragma once

#ifndef _GAPPUTILS_CSVWRITER_H_
#define _GAPPUTILS_CSVWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {

namespace common {

class CsvWriter : public workflow::DefaultWorkflowElement<CsvWriter>
{
  InitReflectableClass(CsvWriter)

  Property(Data, boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > >)
  Property(Filename, std::string)

public:
  CsvWriter(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif
