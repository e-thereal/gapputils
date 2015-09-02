/*
 * ObservableOptimizer.cpp
 *
 *  Created on: Nov 5, 2008
 *      Author: tombr
 */

template<class D>
void ObservableOptimizer<D>::addObserver(IOptimizerObserver<D>& observer) {
  observers.push_back(&observer);
}

template<class D> template<class D2>
void ObservableOptimizer<D>::propagateObservers(IFunction<D2>& function) {
  IFunctionObserver<D2>* observer;
  IObservableFunction<D2>* obsFunc;
  if (!(obsFunc = dynamic_cast<IObservableFunction<D2>*>(&function)))
    return;

  for (unsigned i = 0; i < observers.size(); ++i) {
    if ((observer = dynamic_cast<IFunctionObserver<D2>* >(observers[i]))) {
      obsFunc->addObserver(*observer);
    }
  }
}

template<class D> template<class D2>
void ObservableOptimizer<D>::propagateObservers(IObservableOptimizer<D2>& optimizer) {
  for (unsigned i = 0; i < observers.size(); ++i) {
    propagateObserver(optimizer, *observers[i]);
  }
}

template<class D> template<class D2>
void ObservableOptimizer<D>::propagateObserver(IObservableOptimizer<D2>& optimizer,
    IOptimizerObserver<D>& observer)
{
  IOptimizerObserver<D2>* pobserver;

  if ((pobserver = dynamic_cast<IOptimizerObserver<D2>* >(&observer))) {
    optimizer.addObserver(*pobserver);
  }
}

template<class D>
void ObservableOptimizer<D>::fireEventTriggered(const IOptimizerEvent& event) {
  for (unsigned i = 0; i < observers.size(); ++i) {
    observers[i]->eventTriggered(event, *this);
  }
}
