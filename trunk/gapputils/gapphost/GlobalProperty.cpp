/*
 * GlobalProperty.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "GlobalProperty.h"

#include <capputils/VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(GlobalProperty)

  DefineProperty(Name)
  DefineProperty(ModuleUuid)
  DefineProperty(PropertyId)
  DefineProperty(Edges, Volatile())
  DefineProperty(Expressions, Volatile())

EndPropertyDefinitions

GlobalProperty::GlobalProperty()
 : _Edges(new std::vector<boost::weak_ptr<Edge> >()),
   _Expressions(new std::vector<boost::weak_ptr<Expression> >())
{ }

GlobalProperty::~GlobalProperty() { }

void GlobalProperty::addEdge(boost::shared_ptr<Edge> edge) {
  _Edges->push_back(edge);
}

void GlobalProperty::removeEdge(boost::shared_ptr<Edge> edge) {
  for (unsigned i = 0; i < _Edges->size(); ++i) {
    if (_Edges->at(i).lock() == edge) {
      _Edges->erase(_Edges->begin() + i);
      return;
    }
  }
}

}

}
