/*
 * SupportVectorClassifier.h
 *
 *  Created on: Oct 17, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_ML_SUPPORTVECTORCLASSIFIER_H_
#define GAPPUTLIS_ML_SUPPORTVECTORCLASSIFIER_H_

#include <fstream>
#include <vector>

#include <gpusvm/svmCommon.h>
#include <dlib/matrix.h>

namespace gapputils {

namespace ml {

namespace segmentation {

/**
 * This trainer always uses an RBF kernel
 *
 * TODO: This class will be moved into the gpusvm library as the default interface
 *
 */
class SupportVectorClassifier {
public:
  typedef float value_t;

private:
  int supportVectorCount;
  size_t dimensions;
  gpusvm::Kernel_params kp;

  value_t* supportVectors;
  value_t* alphas;

public:
  SupportVectorClassifier();
  virtual ~SupportVectorClassifier();

  void train(std::vector<value_t>& samples, std::vector<value_t>& labels, value_t gamma, value_t c);

  /// labels must be resized to hold the predicted labels
  void classify(std::vector<value_t>& samples, std::vector<value_t>& labels);

  dlib::matrix<double, 2, 1> validate(std::vector<value_t>& samples, std::vector<value_t>& labels);
  static dlib::matrix<double, 2, 1> CrossValidate(const std::vector<value_t>& samples, const std::vector<value_t>& labels, value_t gamma, value_t c, size_t folds);

  size_t getSupportVectorCount() const;
  size_t getDimensions() const;

  void serialize(std::ofstream& file) const;
  void deserialize(std::ifstream& file);
};

} /* namespace segmentation */

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTLIS_ML_SUPPORTVECTORCLASSIFIER_H_ */
