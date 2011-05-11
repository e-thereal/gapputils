#include "IntSplitter.h"

#include <ObserveAttribute.h>
#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <VolatileAttribute.h>
#include <LabelAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(IntSplitter)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(In, Input(), Observe(PROPERTY_ID))
  DefineProperty(Out1, Output(), Observe(PROPERTY_ID))
  DefineProperty(Out2, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

void changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId != 1)
    return;

  IntSplitter* splitter = dynamic_cast<IntSplitter*>(sender);
  if (splitter) {
    splitter->setOut1(splitter->getIn());
    splitter->setOut2(splitter->getIn());
  }
}

IntSplitter::IntSplitter(void) : _Label("IS"), _In(0), _Out1(0), _Out2(0)
{
  Changed.connect(changeHandler);
}


IntSplitter::~IntSplitter(void)
{
}

BeginPropertyDefinitions(IntSplitter4)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(In, Input(), Observe(PROPERTY_ID))
  DefineProperty(Out1, Output(), Observe(PROPERTY_ID))
  DefineProperty(Out2, Output(), Observe(PROPERTY_ID))
  DefineProperty(Out3, Output(), Observe(PROPERTY_ID))
  DefineProperty(Out4, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

void changeHandler4(capputils::ObservableClass* sender, int eventId) {
  if (eventId != 1)
    return;

  IntSplitter4* splitter = dynamic_cast<IntSplitter4*>(sender);
  if (splitter) {
    splitter->setOut1(splitter->getIn());
    splitter->setOut2(splitter->getIn());
    splitter->setOut3(splitter->getIn());
    splitter->setOut4(splitter->getIn());
  }
}

IntSplitter4::IntSplitter4(void) : _Label("IS4"), _In(0), _Out1(0), _Out2(0), _Out3(0), _Out4(0)
{
  Changed.connect(changeHandler4);
}


IntSplitter4::~IntSplitter4(void)
{
}

}
