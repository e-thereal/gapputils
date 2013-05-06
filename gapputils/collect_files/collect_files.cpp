//============================================================================
// Name        : collect_files.cpp
// Author      : Tom Brosch
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

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

int main(int argc, char** argv) {

  named_mutex mutex(open_or_create, "grapevine_collect_files");
  scoped_lock<named_mutex> lock(mutex);

  managed_shared_memory segment(open_or_create, "grapevine_collect_files_memory", 1 << 16);
  CharAllocator charAllocator (segment.get_segment_manager());
  StringAllocator stringAllocator (segment.get_segment_manager());

  MyVector *fileList = segment.find_or_construct<MyVector>("file_list")(stringAllocator);
  time_t* lastUpdate = segment.find_or_construct<time_t>("last_update")(time(NULL));

  if (*lastUpdate + 1 < time(NULL)) {
    fileList->clear();
  }

  for (int iArg = 1; iArg < argc; ++iArg) {
    MyString str(charAllocator);
    str = argv[iArg];
    fileList->push_back(str);
  }

  *lastUpdate = time(NULL);
	return 0;
}
