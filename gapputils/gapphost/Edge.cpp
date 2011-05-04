#include "Edge.h"

#include <VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(Edge)

  DefineProperty(OutputNode)
  DefineProperty(OutputProperty)
  DefineProperty(InputNode)
  DefineProperty(InputProperty)
  DefineProperty(CableItem, Volatile())

EndPropertyDefinitions

Edge::Edge(void) : _CableItem(0)
{
}

Edge::~Edge(void)
{
}

}

}
