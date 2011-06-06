#include "ToRgb.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ShortNameAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ToRgb)
  ReflectableBase(workflow::WorkflowElement)

  DefineProperty(ImagePtr, Input("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Red, Output("R"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Green, Output("G"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Blue, Output("B"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ToRgb::ToRgb(void) : _ImagePtr(0), _Red(0), _Green(0), _Blue(0), data(0)
{
  setLabel("ToRgb");
  Changed.connect(capputils::EventHandler<ToRgb>(this, &ToRgb::changedEventHandler));
}

ToRgb::~ToRgb(void)
{
  if (data)
    delete data;
}

void ToRgb::changedEventHandler(capputils::ObservableClass* sender, int eventId) {
  
}

void ToRgb::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ToRgb();

  if (!capputils::Verifier::Valid(*this))
    return;

}

void ToRgb::writeResults() {
  
}

}

}