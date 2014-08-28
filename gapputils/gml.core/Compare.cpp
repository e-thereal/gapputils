#include "Compare.h"

#include <cmath>

namespace gml {

namespace core {

BeginPropertyDefinitions(Compare)

  ReflectableBase(DefaultWorkflowElement<Compare>)

  WorkflowProperty(X, Input())
  WorkflowProperty(Y, Input())
  WorkflowProperty(Xs, Input())
  WorkflowProperty(Ys, Input())
  WorkflowProperty(Type, Enumerator<Type>(),
      Description("One of mean squared error (MSE), root mean squared error (RMSE), relative root mean squared error (RRMSE), or classification error of a multinomial distribution (Multinomial)"))
  WorkflowProperty(Error, Output())

EndPropertyDefinitions

Compare::Compare(void) : _Type(ErrorType::MSE), _Error(0) {
  setLabel("Compare");
}

void Compare::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  size_t count = 0;
  double totalError = 0;

  if (getX() && getY()) {
    double error = 0;

    data_t& x = *getX();
    data_t& y = *getY();

    if (x.size() != y.size()) {
      dlog(Severity::Warning) << "X and y must have the same size. Aborting!";
      return;
    }

    switch (getType()) {
    case ErrorType::MSE:
      {
        for (size_t i = 0; i < x.size(); ++i) {
          error += (x[i] - y[i]) * (x[i] - y[i]);
        }
        error /= x.size();
      }
      break;
    case ErrorType::RMSE:
      {
        for (size_t i = 0; i < x.size(); ++i) {
          error += (x[i] - y[i]) * (x[i] - y[i]);
        }
        error /= x.size();
        error = sqrt(error);
      }
      break;
    case ErrorType::RRMSE:
      {
        double xMean = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
          error += (x[i] - y[i]) * (x[i] - y[i]);
          xMean += x[i];
        }
        error /= x.size();
        xMean /= x.size();
        error = sqrt(error);
        error /= xMean;
      }
      break;

    case ErrorType::Multinomial:
      {
        // Get index of maximum value
        int xMax = 0, yMax = 0;
        for (size_t i = 1; i < x.size(); ++i) {
          if (x[i] > x[xMax])
            xMax = i;
          if (y[i] > y[yMax])
            yMax = i;
        }
        error = (xMax != yMax);
      }
      break;
    }
    totalError += error;
    ++count;
  }

  if (getXs() && getYs()) {

    v_data_t& xs = *getXs();
    v_data_t& ys = *getYs();

    for (size_t iSample = 0; iSample < xs.size() && iSample < ys.size(); ++iSample) {
      double error = 0;

      if (!xs[iSample] || !ys[iSample])
        continue;

      data_t& x = *xs[iSample];
      data_t& y = *ys[iSample];

      if (x.size() != y.size()) {
        dlog(Severity::Warning) << "X and y must have the same size. Skipping sample " << iSample + 1 << "!";
        continue;
      }

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

      case ErrorType::Multinomial:
        {
          // Get index of maximum value
          int xMax = 0, yMax = 0;
          for (size_t i = 1; i < x.size(); ++i) {
            if (x[i] > x[xMax])
              xMax = i;
            if (y[i] > y[yMax])
              yMax = i;
          }
          error = (xMax != yMax);
        }
        break;
      }
      totalError += error;
      ++count;
    }
  }

  if (count == 0) {
    dlog(Severity::Warning) << "No input given. Aborting!";
    return;
  }

  newState->setError(totalError / count);
}

}

}
