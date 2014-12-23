/**
 * @file TypeSystem.h
 * @brief Basic algebraic types
 *
 * @date Nov 7, 2008
 * @author Tom Brosch
 */

#ifndef _ALGLIB_TYPESYSTEM_H_
#define _ALGLIB_TYPESYSTEM_H_

#include <vector>

#include "alglib.h"

#if defined(_MSC_VER) /* MSVC Compiler */
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif

/// Basic definitions related to algebra
/** The alglib namespace defines basic algebraic structures and algorithms like
 * what a function in general is.
 */
namespace alglib {

/// Common base for all functions
class IFunctionBase {
public:
  /// Virtual destructor
  virtual ~IFunctionBase() { }
};

/// Defines our understanding of a function in general.
/**
 * A function describes the mapping of an element of the function
 * domain (D) to an element of the function range (R)
 *
 * @remarks
 *
 *  - We want to use IFunction as an interface, hence always inherit
 *    from IFunction as a virtual base class.
 *    @code
 *    class MyFunction : public virtual IFunction { ... };
 *    @endcode
 *    This prevents ambiguities that might occur due to multiple inheritance!
 */
template<class D, class R>
class IFunction : public virtual IFunctionBase {
public:
  typedef D DomainType; ///< Makes the DomainType of the function accessible
  typedef R RangeType;  ///< Makes the RangeType of the function accessible

public:
  /// Overload this method to define the function
  /**
   * @param[in] parameter The function parameter of DomainType
   *
   * @return The function value of RangeType
   */
  virtual RangeType eval(const DomainType& parameter) = 0;
};

/// Use this type to define coordinates
typedef std::vector<double> Coordinates;

/// Use this type in case of discrete coordinates
typedef std::vector<int> DiscreteCoordinates;

/// Defines an arbitrary key value pair
template<class K, class V>
struct Pair {
  typedef K KeyType;    ///< The key type of the pair
  typedef V ValueType;  ///< The value type of the pair

  /// The key or x value
  union {
    K x;      ///< The x value
    K key;    ///< The key
  };

  /// The value or y value
  union {
    V y;      ///< The y value
    V value;  ///< The value
  };

  /// Constructor to create a pair
  Pair(K x, V y) : x(x), y(y) { }
};

}

#if defined(_MSC_VER) /* MSVC Compiler */
#pragma warning( pop ) 
#endif

#endif /* _ALGLIB_TYPESYSTEM_H_ */
