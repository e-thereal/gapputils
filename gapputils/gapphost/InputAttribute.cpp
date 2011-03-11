#include "InputAttribute.h"

using namespace capputils::attributes;

namespace gapputils {

namespace attributes {

InputAttribute::InputAttribute(void)
{
}


InputAttribute::~InputAttribute(void)
{
}

AttributeWrapper* Input() {
  return new AttributeWrapper(new InputAttribute());
}

}

}
