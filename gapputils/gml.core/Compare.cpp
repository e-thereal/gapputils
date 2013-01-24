#include "Compare.h"

#include <cmath>

namespace gml {

namespace core {

BeginPropertyDefinitions(Compare)

  ReflectableBase(DefaultWorkflowElement<Compare>)

  WorkflowProperty(X, Input(), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Y, Input(), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Type, Enumerator<Type>(),
      Description("One of mean squared error (MSE), root mean squared error (RMSE), or relative root mean squared error (RRMSE)."))
  WorkflowProperty(Error, Output())

EndPropertyDefinitions

Compare::Compare(void) : _Type(ErrorType::MSE), _Error(0) {
  setLabel("Compare");
}

void Compare::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::vector<double>& x = *getX();
  std::vector<double>& y = *getY();

  if (x.size() != y.size()) {
    dlog(Severity::Warning) << "X and y must have the same size. Aborting!";
    return;
  }

  double error = 0;
  switch (getType()) {
  case ErrorType::MSE: {
      for (size_t i = 0; i < x.size(); ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
      }
      error /= x.size();
    } break;
  case ErrorType::RMSE: {
      for (size_t i = 0; i < x.size(); ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
      }
      error /= x.size();
      error = sqrt(error);
    } break;
  case ErrorType::RRMSE: {
      double xMean = 0.0;
      for (size_t i = 0; i < x.size(); ++i) {
        error += (x[i] - y[i]) * (x[i] - y[i]);
        xMean += x[i];
      }
      error /= x.size();
      xMean /= x.size();
      error = sqrt(error);
      error /= xMean;
    } break;
  }
  newState->setError(error);
}

}

}
