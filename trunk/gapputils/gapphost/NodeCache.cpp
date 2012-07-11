/*
 * NodeCache.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */
#define BOOST_FILESYSTEM_VERSION 2
#include "NodeCache.h"

#include <capputils/ReflectableClass.h>
#include <capputils/Serializer.h>
#include <capputils/SerializeAttribute.h>
#include <capputils/NoParameterAttribute.h>

#include <gapputils/CacheableAttribute.h>
#include <gapputils/LabelAttribute.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#ifdef GAPPHOST_HAVE_ZLIB
#include <boost/iostreams/filter/gzip.hpp>
#endif
#include <boost/iostreams/device/file_descriptor.hpp>

#include "Node.h"
#include "ChecksumUpdater.h"

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;
using namespace gapputils::attributes;

namespace bio = boost::iostreams;

namespace gapputils {

namespace host {

NodeCache::NodeCache() {
}

NodeCache::~NodeCache() {
}

void NodeCache::Update(workflow::Node* node) {
  if (!node)
    return;

  ReflectableClass* module = node->getModule();

  if (!module)
    return;

  // Cache only if module has the gapputils::attributes::CacheModuleAttribute
  // Confirm that all non-parameters (except inputs) are serializable
  // Calculate checksum over all inputs and all parameters
  // Open cache file (.gapphost/cache/<Uuid>/<checksum>.cache)
  // Serialize module to cache file
  // Append final checksum over all properties of the module

  if (!module->getAttribute<CacheableAttribute>())
    return;


  std::vector<IClassProperty*>& properties = module->getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty* prop = properties[i];
    if (prop->getAttribute<NoParameterAttribute>() &&
        !prop->getAttribute<ISerializeAttribute>() &&
        !prop->getAttribute<LabelAttribute>())
    {
      std::cout << "[Warning] Can't cache module '" << node->getUuid() << "' because property '"
                << prop->getName() << "' is not serializable." << std::endl;
      return;
    }
  }

  std::string cacheDirectory = ".gapphost/cache/" + node->getUuid();
  std::stringstream cacheName;
  cacheName << cacheDirectory << "/"
            << ChecksumUpdater::GetChecksum(node, ChecksumUpdater::ExcludeNoParameters)
            << ".cache";

  boost::filesystem::create_directories(cacheDirectory);
  bio::filtering_ostream cacheFile;
#ifdef GAPPHOST_HAVE_ZLIB
  cacheFile.push(boost::iostreams::gzip_compressor());
#endif
  cacheFile.push(bio::file_descriptor_sink(cacheName.str().c_str()));
  if (!cacheFile)
    return;

  Serializer::writeToFile(*module, cacheFile);

  checksum_t checksum = ChecksumUpdater::GetChecksum(node, ChecksumUpdater::NoExclude);
  cacheFile.write((char*)&checksum, sizeof(checksum));
}

bool NodeCache::Restore(workflow::Node* node) {
  assert(node);

  ReflectableClass* module = node->getModule();

  if (!module)
    return false;

  // Calculate checksum over all inputs and all parameters
  // Open cache file and restore module if possible
  // Calculate checksum over all properties
  // Compare checksum with cache checksum

  if (!module->getAttribute<CacheableAttribute>())
    return false;

  std::stringstream cacheName;
  cacheName << ".gapphost/cache/" << node->getUuid() << "/"
            << ChecksumUpdater::GetChecksum(node, ChecksumUpdater::ExcludeNoParameters)
            << ".cache";
  if (!boost::filesystem::exists(cacheName.str())) {
    std::cout << "[Info] No cache for module '" << node->getUuid() << "'." << std::endl;
    return false;
  }

  boost::iostreams::filtering_istream cacheFile;
#ifdef GAPPHOST_HAVE_ZLIB
  cacheFile.push(boost::iostreams::gzip_decompressor());
#endif
  cacheFile.push(bio::file_descriptor_source(cacheName.str().c_str()));
  if (!cacheFile) {
    std::cout << "[Warning] Can't open cache file for module '" << node->getUuid() << "'." << std::endl;
    return false;
  }

  Serializer::readFromFile(*module, cacheFile);


  checksum_t checksum, currentChecksum = ChecksumUpdater::GetChecksum(node, ChecksumUpdater::NoExclude);
  cacheFile.read((char*)&checksum, sizeof(checksum));

  if (cacheFile.bad()) {
    std::cout << "[Info] Can't read checksum for module '" << node->getUuid() << "'." << std::endl;
    return false;
  }

  if (currentChecksum != checksum) {
    std::cout << "[Info] Checksums don't match for module '" << node->getUuid() << "'." << std::endl;
    return false;
  }
  std::cout << "[Info] Reading value from cache for module '" << node->getUuid() << "'." << std::endl;

  return true;
}

}

}
