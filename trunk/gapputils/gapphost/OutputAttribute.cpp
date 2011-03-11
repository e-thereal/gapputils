#include "OutputAttribute.h"

using namespace capputils::attributes;

namespace gapputils {

namespace attributes {

OutputAttribute::OutputAttribute(void)
{
}


OutputAttribute::~OutputAttribute(void)
{
}

AttributeWrapper* Output() {
  return new AttributeWrapper(new OutputAttribute());
}

}

}
