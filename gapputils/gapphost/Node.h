#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <boost/crc.hpp>

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <gapputils/IProgressMonitor.h>
#include "ModelHarmonizer.h"

namespace gapputils {

class ToolItem;

namespace workflow {

class Node : public capputils::reflection::ReflectableClass,
             public capputils::ObservableClass
{
public:
  typedef boost::crc_32_type::value_type checksum_type;

  InitReflectableClass(Node)

  Property(Uuid, std::string)
  Property(X, int)
  Property(Y, int)
  Property(Module, capputils::reflection::ReflectableClass*)
  Property(InputChecksum, checksum_type)
  Property(OutputChecksum, checksum_type)
  Property(ToolItem, ToolItem*)

private:
  ModelHarmonizer* harmonizer;
  static int moduleId;

public:
  Node();
  virtual ~Node(void);

  virtual bool isUpToDate() const;
  virtual void update(IProgressMonitor* monitor);
  virtual void writeResults();
  virtual void resume();

  QStandardItemModel* getModel();

  /**
   * \brief Updates the checksum of the current node
   *
   * \param[in] inputChecksums  String containing the concatenation of all direct input checksums
   *
   * \remark
   * - Workflows overload this method in order to update the checksum of all nodes first before
   *   calculating the input checksum
   */
  virtual void updateChecksum(const std::vector<checksum_type>& inputChecksums);

  static checksum_type getChecksum(const capputils::reflection::IClassProperty* property,
      const capputils::reflection::ReflectableClass& object);

  /**
   * \brief Caches the state of the module if possible
   *
   * Caching is only possible if all non-parameters (except inputs) are serializable
   */
  virtual void updateCache();

  /**
   * \brief Tries to restore the state of a module from the module cache
   *
   * \return True, iff the state could be sucessfully restored from the cache
   */
  virtual bool restoreFromCache();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
