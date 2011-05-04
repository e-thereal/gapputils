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

enum EventIds {
  TypeId, XId, YId, CountId, ErrorId
};

using namespace attributes;

BeginPropertyDefinitions(Compare)

  ReflectableProperty(Type, Observe(TypeId), Label())
  DefineProperty(X, Observe(XId), Input(), NotEqual<double*>(0), Hide(), Volatile())
  DefineProperty(Y, Observe(YId), Input(), NotEqual<double*>(0), Hide(), Volatile())
  DefineProperty(Count, Observe(CountId), Input())
  DefineProperty(Error, Observe(ErrorId), Output())

EndPropertyDefinitions

Compare::Compare(void) : _Type(ErrorType::MSE), _X(0), _Y(0), _Count(0), _Error(0)
{
  Changed.connect(capputils::EventHandler<Compare>(this, &Compare::changeEventHandler));
}


Compare::~Compare(void)
{
}

void Compare::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
  if (!capputils::Verifier::Valid(*this))
    return;

  if (eventId == XId || eventId == YId || eventId == TypeId) {
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
    setError(error);
  }
}

}
