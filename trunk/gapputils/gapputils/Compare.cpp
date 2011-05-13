#include "Compare.h"

#include <ObserveAttribute.h>
#include "InputAttribute.h"
#include "OutputAttribute.h"
#include "LabelAttribute.h"
#include <NotEqualAssertion.h>
#include <Verifier.h>
#include <EventHandler.h>

#include "HideAttribute.h"
#include <VolatileAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(Compare)

  ReflectableProperty(Type, Observe(PROPERTY_ID), Label())
  DefineProperty(X, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile())
  DefineProperty(Y, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile())
  DefineProperty(Count, Observe(PROPERTY_ID), Input("N"))
  DefineProperty(Error, Observe(PROPERTY_ID), Output())

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
