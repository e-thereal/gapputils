/*
 * NodeCache.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_NODECACHE_H_
#define GAPPUTILS_HOST_NODECACHE_H_

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace workflow {

class Node;

}

namespace host {

class NodeCache {
public:
  NodeCache();
  virtual ~NodeCache();

  /**
   * \brief Caches the state of the module if possible
   *
   * Caching is only possible if all non-parameters (except inputs) are serializable
   */
  static void Update(boost::shared_ptr<workflow::Node> node);

  /**
   * \brief Tries to restore the state of a module from the module cache
   *
   * \return True, iff the state could be successfully restored from the cache
   */
  static bool Restore(boost::shared_ptr<workflow::Node> node);
};

}

}
#endif /* GAPPUTILS_HOST_NODECACHE_H_ */
