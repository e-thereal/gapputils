#pragma once

#ifndef GAPPUTILS_HOST_CHECKSUMUPDATER_H_
#define GAPPUTILS_HOST_CHECKSUMUPDATER_H_

#include <stack>

#include <capputils/reflection/ReflectableClass.h>
#include <gapputils/gapputils.h>
#include <boost/weak_ptr.hpp>

namespace gapputils {

namespace workflow {
class Node;
}

namespace host {

/// Updates checksums
class ChecksumUpdater
{
public:
  enum ChecksumFlags {
    NoExclude = 0,
    ExcludeNoParameters = 1,
    ExcludeDependent = 2,
    OnlyNondependentParameters = ExcludeNoParameters | ExcludeDependent
  };
private:
  std::stack<boost::shared_ptr<workflow::Node> > nodesStack;

public:
  ChecksumUpdater(void);
  ~ChecksumUpdater(void);

  /// Updates the checksum of the given node and all dependent nodes
  /**
   * If the given node is a workflow, the checksums of all output nodes of the workflow are updated as well
   * The workflow checksum is then build from the checksums of the output nodes
   */
  void update(boost::shared_ptr<workflow::Node> node);

  static checksum_t GetChecksum(boost::shared_ptr<workflow::Node> node, int flags = OnlyNondependentParameters);

private:
  void buildStack(boost::shared_ptr<workflow::Node> node);
};

}

}
#endif
