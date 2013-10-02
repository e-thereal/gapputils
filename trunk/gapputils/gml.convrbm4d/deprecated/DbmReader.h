/*
 * DbmReader.h
 *
 *  Created on: Jul 09, 2013
 *      Author: tombr
 */

#ifndef GML_DBMREADER_H_
#define GML_DBMREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "DbmModel.h"

namespace gml {
namespace convrbm4d {

class DbmReader : public DefaultWorkflowElement<DbmReader> {
  InitReflectableClass(DbmReader)

  Property(Filename, std::string)
  Property(Model, boost::shared_ptr<DbmModel>)
  Property(FilterWidth, std::vector<int>)
  Property(FilterHeight, std::vector<int>)
  Property(FilterDepth, std::vector<int>)
  Property(ChannelCount, std::vector<int>)
  Property(FilterCount, std::vector<int>)

public:
  DbmReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_DBMREADER_H_ */
