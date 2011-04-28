#include "HideAttribute.h"

using namespace capputils::attributes;

namespace gapputils {

namespace attributes {

HideAttribute::HideAttribute(void)
{
}


HideAttribute::~HideAttribute(void)
{
}

AttributeWrapper* Hide() {
  return new AttributeWrapper(new HideAttribute());
}

}

}
