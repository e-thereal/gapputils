/*
 * GlobalProperty.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#include "GlobalProperty.h"

#include "GlobalEdge.h"

#include <capputils/VolatileAttribute.h>
#include <boost/regex.hpp>

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
 : _Edges(new std::vector<boost::weak_ptr<GlobalEdge> >()),
   _Expressions(new std::vector<boost::weak_ptr<Expression> >())
{ }

GlobalProperty::~GlobalProperty() { }

void GlobalProperty::addEdge(boost::shared_ptr<GlobalEdge> edge) {
  _Edges->push_back(edge);
}

void GlobalProperty::removeEdge(boost::shared_ptr<GlobalEdge> edge) {
  for (unsigned i = 0; i < _Edges->size(); ++i) {
    if (_Edges->at(i).lock() == edge) {
      _Edges->erase(_Edges->begin() + i);
      return;
    }
  }
}

void GlobalProperty::rename(const std::string& name) {
  for (size_t i = 0; i < _Edges->size(); ++i) {
    boost::shared_ptr<GlobalEdge> edge = _Edges->at(i).lock();
    edge->setGlobalProperty(name);
  }

  std::string oldTag = std::string("\\$\\(") + getName() + "\\)";
  std::string newTag = std::string("$(") + name + ")";

  setName(name);

  for (size_t i = 0; i < _Expressions->size(); ++i) {
    boost::shared_ptr<Expression> expr = _Expressions->at(i).lock();
    std::string exprStr = expr->getExpression();
    std::string exprStr2 = boost::regex_replace(exprStr, boost::regex(oldTag), newTag);
    std::cout << "Renaming expressin from " << exprStr << " to " << exprStr2 << std::endl;
    expr->setExpression(exprStr2);
  }
}

}

}
