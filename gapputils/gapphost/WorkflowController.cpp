#define BOOST_FILESYSTEM_VERSION 2

#include "WorkflowController.h"

#include <capputils/FilenameAttribute.h>
#include <capputils/IEnumerableAttribute.h>
#include <capputils/IReflectableAttribute.h>
#include <capputils/ScalarAttribute.h>

#include <gapputils/LabelAttribute.h>
#include <gapputils/ChecksumAttribute.h>

#include <boost/filesystem.hpp>

using namespace capputils;
using namespace capputils::attributes;
using namespace capputils::reflection;

namespace gapputils {

using namespace attributes;

namespace host {

WorkflowController::WorkflowController(void)
{
}


WorkflowController::~WorkflowController(void)
{
}

}

}
