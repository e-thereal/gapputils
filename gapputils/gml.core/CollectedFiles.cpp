/*
 * CollectedFiles.cpp
 *
 *  Created on: May 6, 2013
 *      Author: tombr
 */

#include "CollectedFiles.h"

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost::interprocess;

typedef allocator<char, managed_shared_memory::segment_manager>  CharAllocator;
typedef basic_string<char, std::char_traits<char>, CharAllocator> MyString;
typedef allocator<MyString, managed_shared_memory::segment_manager>  StringAllocator;
typedef vector<MyString, StringAllocator> MyVector;

namespace gml {

namespace core {

BeginPropertyDefinitions(CollectedFiles)

  ReflectableBase(DefaultWorkflowElement<CollectedFiles>)

  WorkflowProperty(Filenames, Filename("", true), FileExists(), Enumerable<Type, false>())
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

CollectedFiles::CollectedFiles() {
  setLabel("Files");
}

void CollectedFiles::show() {
  named_mutex mutex(open_or_create, "grapevine_collect_files");
  scoped_lock<named_mutex> lock(mutex);

  managed_shared_memory segment(open_or_create, "grapevine_collect_files_memory", 1 << 16);
  CharAllocator charAllocator (segment.get_segment_manager());
  StringAllocator stringAllocator (segment.get_segment_manager());

  MyVector *fileList = segment.find_or_construct<MyVector>("file_list")(stringAllocator);

  std::vector<std::string> list;
  for (size_t i = 0; i < fileList->size(); ++i) {
    list.push_back(fileList->at(i).c_str());
  }
  setFilenames(list);

  execute(0);
  writeResults();
}

void CollectedFiles::update(IProgressMonitor* monitor) const {
  newState->setOutput(getFilenames());
}

} /* namespace core */

} /* namespace gml */
