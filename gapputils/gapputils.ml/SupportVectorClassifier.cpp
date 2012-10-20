/*
 * SvmSegmenter.cpp
 *
 *  Created on: Oct 17, 2012
 *      Author: tombr
 */

#include "SupportVectorClassifier.h"

#include <gpusvm/svmTrain.h>
#include <gpusvm/svmClassify.h>

#include <algorithm>
#include <iostream>

#include <cassert>
#include <cstdlib>

namespace gapputils {

namespace ml {

namespace segmentation {

typedef SupportVectorClassifier::value_t value_t;

SupportVectorClassifier::SupportVectorClassifier() : supportVectorCount(0), dimensions(0), supportVectors(0), alphas(0) {
  kp.kernel_type = "rbf";
}

SupportVectorClassifier::~SupportVectorClassifier() {
  if (supportVectors)
    free(supportVectors);

  if (alphas)
    free(alphas);
}

void SupportVectorClassifier::train(std::vector<value_t>& samples, std::vector<value_t>& labels, value_t gamma, value_t c) {
  float* trainingAlphas;

  kp.gamma = gamma;
  dimensions = samples.size() / labels.size();

  if (supportVectors)
    free(supportVectors);

  if (alphas)
    free(alphas);

  gpusvm::performTraining(&samples[0], labels.size(), dimensions, &labels[0], &trainingAlphas, &kp, c);
  gpusvm::formModel(&samples[0], labels.size(), dimensions, trainingAlphas,
      &labels[0], &supportVectors, &supportVectorCount, &alphas);

  free(trainingAlphas);
}

void SupportVectorClassifier::classify(std::vector<value_t>& samples, std::vector<value_t>& labels) {
  assert(samples.size() / labels.size() == dimensions);

  float* predictedLabels;

  gpusvm::performClassification(&samples[0], labels.size(),
      supportVectors, supportVectorCount, dimensions,
      alphas, kp, &predictedLabels);

  std::copy(predictedLabels, predictedLabels + labels.size(), labels.begin());

  free(predictedLabels);
}

dlib::matrix<double, 2, 1> SupportVectorClassifier::validate(std::vector<value_t>& samples, std::vector<value_t>& labels) {
  double TP = 0, TN = 0, NP = 0, NN = 0;
  std::vector<value_t> predictedLabels(labels.size());
  classify(samples, predictedLabels);

  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0) {
      ++NP;
      if (predictedLabels[i] > 0)
        ++TP;
    } else {
      ++NN;
      if (predictedLabels[i] < 0)
        ++TN;
    }
  }
  dlib::matrix<double, 2, 1> result;
  result = TP / NP, TN / NN;
  std::cout << "TP: " << TP << "; NP: " << NP << "; TN: " << TN << "; NN: " << NN << std::endl;
  return result;
}

void create_fold(const std::vector<value_t>& samples, const std::vector<value_t>& labels, size_t iFold, size_t foldCount,
    std::vector<value_t>& trainingSamples, std::vector<value_t>& trainingLabels,
    std::vector<value_t>& testSamples, std::vector<value_t>& testLabels)
{
  const size_t sampleCount = labels.size();
  const size_t featureCount = samples.size() / sampleCount;
  const size_t samplesPerFold = sampleCount / foldCount;
  const size_t usedSamples = foldCount * samplesPerFold;

  trainingSamples.clear();
  trainingLabels.clear();
  testSamples.clear();
  testLabels.clear();

  for (size_t j = 0; j < featureCount; ++j) {
    for (size_t i = 0; i < usedSamples; ++i) {
      if (i / samplesPerFold == iFold)
        testSamples.push_back(samples[i + j * sampleCount]);
      else
        trainingSamples.push_back(samples[i + j * sampleCount]);
    }
  }
  for (size_t i = 0; i < usedSamples; ++i) {
    if (i / samplesPerFold == iFold)
      testLabels.push_back(labels[i]);
    else
      trainingLabels.push_back(labels[i]);
  }
}

dlib::matrix<double, 2, 1> SupportVectorClassifier::CrossValidate(
    const std::vector<value_t>& samples, const std::vector<value_t>& labels,
    value_t gamma, value_t c, size_t folds)
{
  std::vector<value_t> trainingSamples, trainingLabels, testSamples, testLabels;

  dlib::matrix<double, 2, 1> result;
  result = 0, 0;
  for (size_t iFold = 0; iFold < folds; ++iFold) {
    create_fold(samples, labels, iFold, folds, trainingSamples, trainingLabels, testSamples, testLabels);

    SupportVectorClassifier svc;
    svc.train(trainingSamples, trainingLabels, gamma, c);
    result += svc.validate(testSamples, testLabels) / (double)folds;
  }

  return result;
}

size_t SupportVectorClassifier::getSupportVectorCount() const {
  return supportVectorCount;
}

size_t SupportVectorClassifier::getDimensions() const {
  return dimensions;
}

void SupportVectorClassifier::serialize(std::ofstream& file) const {
  uint32_t supportVectorCount = this->supportVectorCount;
  uint32_t dimensions = this->dimensions;
  value_t gamma = kp.gamma;
  value_t b = kp.b;

  file.write((char*)&supportVectorCount, sizeof(supportVectorCount));
  file.write((char*)&dimensions, sizeof(dimensions));
  file.write((char*)&gamma, sizeof(gamma));
  file.write((char*)&b, sizeof(b));
  file.write((char*)supportVectors, sizeof(value_t) * supportVectorCount * dimensions);
  file.write((char*)alphas, sizeof(value_t) * supportVectorCount);
}

void SupportVectorClassifier::deserialize(std::ifstream& file) {
  if (supportVectors)
    free(supportVectors);

  if (alphas)
    free(alphas);

  uint32_t supportVectorCount;
  uint32_t dimensions;
  value_t gamma;
  value_t b;

  file.read((char*)&supportVectorCount, sizeof(supportVectorCount));
  file.read((char*)&dimensions, sizeof(dimensions));
  file.read((char*)&gamma, sizeof(gamma));
  file.read((char*)&b, sizeof(b));

  this->supportVectorCount = supportVectorCount;
  this->dimensions = dimensions;
  kp.gamma = gamma;
  kp.b = b;

  supportVectors = new value_t[supportVectorCount * dimensions];
  file.read((char*)supportVectors, sizeof(value_t) * supportVectorCount * dimensions);

  alphas = new value_t[supportVectorCount];
  file.read((char*)alphas, sizeof(value_t) * supportVectorCount);
}

} /* namespace segmentation */

} /* namespace ml */

} /* namespace gapputils */
