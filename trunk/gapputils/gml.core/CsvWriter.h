#pragma once

#ifndef GML_CSVWRITER_H_
#define GML_CSVWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class CsvWriter : public DefaultWorkflowElement<CsvWriter> {

  typedef double value_t;
  typedef std::vector<value_t> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;
  typedef std::vector<boost::shared_ptr<v_data_t> > vv_data_t;

  InitReflectableClass(CsvWriter)

  Property(Data, boost::shared_ptr<vv_data_t>)
  Property(FlatData, boost::shared_ptr<v_data_t>)
  Property(OnlyRowNames, bool)
  Property(RowNames, boost::shared_ptr<std::vector<std::string> >)
  Property(Header, std::string)
  Property(ColumnCount, std::vector<int>)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  CsvWriter(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif
