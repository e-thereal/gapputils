#include "Compare.h"

#include <capputils/EventHandler.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/LabelAttribute.h>
#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

DefineEnum(ErrorType)

BeginPropertyDefinitions(Compare)

  ReflectableProperty(Type, Observe(PROPERTY_ID), Label())
  DefineProperty(X, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Y, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Count, Observe(PROPERTY_ID), Input("N"), TimeStamp(PROPERTY_ID))
  DefineProperty(Error, Observe(PROPERTY_ID), Output(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Compare::Compare(void) : _Type(ErrorType::MSE), _X(0), _Y(0), _Count(0), _Error(0), data(0)
{
}


Compare::~Compare(void)
{
  if (data)
    delete data;
}

void Compare::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Compare();

  if (!capputils::Verifier::Valid(*this))
    return;

  double error = 0;
  switch (getType()) {
  case ErrorType::MSE: {
      double *x = getX();
      double *y = getY();
      int count = getCount();
      for (int i = 0; i < count; ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
      }
      error /= count;
    } break;
  }
  data->setError(error);
}

void Compare::writeResults() {
  if (!data)
    return;
  setError(data->getError());
}

}

}
