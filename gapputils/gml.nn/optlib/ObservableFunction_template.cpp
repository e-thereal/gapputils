/*
 * ObservableFunction_template.cpp
 *
 *  Created on: Nov 5, 2008
 *      Author: tombr
 */

template<class F>
void ObservableFunction<F>::addObserver(IFunctionObserver<typename F::DomainType>& observer) {
  observers.push_back(&observer);
}

template<class F>
void ObservableFunction<F>::fireEvaluationPerformed(const typename F::DomainType& parameter,
    const double& result)
{
  for (unsigned i = 0; i < observers.size(); ++i) {
    observers[i]->evaluationPerformed(parameter, result, *this);
  }
}

template<class F>
double ObservableFunction<F>::eval(const typename F::DomainType& parameter) {
  double result = F::eval(parameter);
  fireEvaluationPerformed(parameter, result);
  return result;
}
