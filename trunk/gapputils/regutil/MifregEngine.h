/**
 * @file MifregEngine.h
 * @brief Contains the MifregEngine
 *
 * @date Nov 25, 2008
 * @author Tom Brosch
 */

#ifndef _REGUTIL_MIFREGENGINE_H_
#define _REGUTIL_MIFREGENGINE_H_

#include <ostream>
#include <string>
#include <vector>

//#include "MifregUI.h"

#include <optlib/ObservableDirectionSetOptimizer.h>

namespace regutil {

class ICudaImage;

/// Easy to use class to register MIF files
/**
 *
 */
class MifregEngine {
public:
  typedef optlib::ObservableDirectionSetOptimizer::ObserverType ObserverType;

  /// List of available optimization algorithms
  enum OptimizationAlgorithm {
    ConjugateGradients,                   ///< The Conjugate Gradients Method with Central Differences
    FastConjugateGradients,               ///< The Conjugate Gradients Method with Forward Differences
    SimplifiedPowell,                     ///< The Simplified Version of Powell's Method
    SteepestDescent                       ///< The Steepest Descent Method
  };
  
  enum SimilarityMeasure {
    OldMethod,
    NewMethod,
    ShamsMethod1,
    ShamsMethod2
  };

private:
  ICudaImage *baseCMif,                   ///< Pointer to the base image
             *inputCMif;                  ///< Pointer to the floating image
  std::vector<ObserverType*> observers;   ///< Vector of observers
  OptimizationAlgorithm algorithm;        ///< The used optimization algorithm
  SimilarityMeasure simMeasure;

public:

  /// Constructor to create a new MifregEngine instance
  MifregEngine();

  /// Adds an observer to the MifregEngine
  /**
   * @remarks
   * - All observers will be propagated to the optimization algorithm
   *
   * @param[in] observer The observer
   */
  void addObserver(ObserverType& observer);

  /// Removes an observer from the MifregEngine
  /**
   * @param[in] observer The observer
   */
  void removeObserver(ObserverType& observer);

  /// Sets the base image
  /**
   * @param[in] cmif A pointer to the base image used for the registration
   */
  void setBaseCMif(ICudaImage *cmif);

  /// Sets the floating image
  /**
   * @param[in] cmif A pointer to the floating image used for the registration
   */
  void setInputCMif(ICudaImage *cmif);

  /// Sets the optimization algorithm, used to solve the registration problem
  /**
   * @param[in] algorithm The optimization algorithm
   */
  void setAlgorithm(OptimizationAlgorithm algorithm);

  /// Gets the currently set optimization algorithm
  /**
   * @return The current optimizatin algorithm
   */
  OptimizationAlgorithm getAlgorithm();
  
  void setSimilarityMeasure(SimilarityMeasure simMeasure);
  SimilarityMeasure getSimilarityMeasure();

  /// Performs the registration (calculates the registration parameters)
  /**
   * @remarks
   * - Use can use initializeResult() to initialize the parameter vector with
   *   a default starting point
   *
   * @param[in,out] result The the initial value of the registration and the registration result
   *
   */
  void performRegistration(optlib::IDirectionSetOptimizer::DomainType& result);

  /// Initializes a parameter vector with a default starting point
  /**
   * @remarks
   * - This method sets all translation and rotation parameters to zero and all
   *   scaling parameters to one.
   *
   * @param[out] result The parameter vector, that should be initialized.
   */
  static void initializeResult(optlib::IDirectionSetOptimizer::DomainType& result);
};

}

/// Print an OptimizationAlgorithm enumeration to an output stream
std::ostream& operator<<(std::ostream& os,
    const regutil::MifregEngine::OptimizationAlgorithm& algorithm);
    
std::ostream& operator<<(std::ostream& os,
    const regutil::MifregEngine::SimilarityMeasure& simMeasure);

#endif /* _REGUTIL_MIFREGENGINE_H_ */
