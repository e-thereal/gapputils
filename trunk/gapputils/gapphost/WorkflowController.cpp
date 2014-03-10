#define BOOST_FILESYSTEM_VERSION 2

#include "WorkflowController.h"

#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/IEnumerableAttribute.h>
#include <capputils/attributes/IReflectableAttribute.h>
#include <capputils/attributes/ScalarAttribute.h>

#include <gapputils/attributes/LabelAttribute.h>
#include <gapputils/attributes/ChecksumAttribute.h>

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
