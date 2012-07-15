#include "Compare.h"

#include <capputils/EnumeratorAttribute.h>
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

#include <iostream>
#include <cmath>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(Compare)

  DefineProperty(Type, Enumerator<ErrorType>(), Observe(PROPERTY_ID), Label())
  DefineProperty(X, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Y, Observe(PROPERTY_ID), Input(), NotEqual<double*>(0), Hide(), Volatile(), TimeStamp(PROPERTY_ID))
  DefineProperty(Count, Observe(PROPERTY_ID), Input("N"), TimeStamp(PROPERTY_ID))
  DefineProperty(Error, Observe(PROPERTY_ID), Output(), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Compare::Compare(void) : _Type(ErrorType::MSE), _X(0), _Y(0), _Count(0), _Error(0)
{
  Changed.connect(capputils::EventHandler<Compare>(this, &Compare::changeEventHandler));
}

Compare::~Compare(void) { }

void Compare::changeEventHandler(capputils::ObservableClass* sender, int eventId) {
}

void Compare::update(gapputils::workflow::IProgressMonitor* monitor) const {
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
  case ErrorType::SE: {
      double *x = getX();
      double *y = getY();
      int count = getCount();
      for (int i = 0; i < count; ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
      }
      error /= count;
      error = sqrt(error);
    } break;
  case ErrorType::RSE: {
      double *x = getX();
      double *y = getY();
      int count = getCount();
      double xMean = 0.0;
      for (int i = 0; i < count; ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
        xMean += x[i];
      }
      error /= count;
      xMean /= count;
      error = sqrt(error);
      error /= xMean;
    } break;
  }
  newState->setError(error);
}

}

}
