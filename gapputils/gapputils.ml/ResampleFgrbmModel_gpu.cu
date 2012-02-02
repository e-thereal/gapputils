/*
 * ResampleFgrbmModel_gpu.cu
 *
 *  Created on: Feb 1, 2012
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT

#include "ResampleFgrbmModel.h"

#include <capputils/Verifier.h>

#include <iostream>

#include <culib/CudaImage.h>
#include <culib/transform.h>
#include <thrust/copy.h>

namespace gapputils {

namespace ml {

/*
  Property(VisibleBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(HiddenBiases, boost::shared_ptr<tbblas::device_vector<double> >)
  Property(VisibleWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(HiddenWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(ConditionalWeights, boost::shared_ptr<tbblas::device_matrix<double> >)
  Property(VisibleMean, double)
  Property(VisibleStd, double)
  Property(IsGaussian, bool)
*/

void ResampleFgrbmModel::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ResampleFgrbmModel();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputModel()) {
    std::cout << "[Warning] No input model given." << std::endl;
    return;
  }

  FgrbmModel& input = *getInputModel();
  boost::shared_ptr<FgrbmModel> output(new FgrbmModel());

  culib::CudaImage inputImage(dim3(getInputWidth(), getInputHeight()));
  culib::CudaImage outputImage(dim3(getOutputWidth(), getOutputHeight()));
  const int inputCount = getInputWidth() * getInputHeight();

  if (inputCount != input.getVisibleBiases()->size()) {
    std::cout << "[Warning] Input image dimension does not match the number of visible units." << std::endl;
    return;
  }

  const int outputCount = getOutputWidth() * getOutputHeight();
  const int featureCount = input.getConditionalWeights()->size2();

  fmatrix4 scalingMatrix = make_fmatrix4_scaling(
      (float)inputImage.getSize().x / (float)outputImage.getSize().x,
      (float)inputImage.getSize().y / (float)outputImage.getSize().y,
      1.f
  );

  tbblas::device_vector<double>& inputVB = *input.getVisibleBiases();
  thrust::copy(inputVB.data().begin(), inputVB.data().end(), thrust::device_ptr<float>(inputImage.getDevicePointer()));
  inputImage.saveDeviceToWorkingCopy();   // I need to do this 'cause getCudaArray() reads from the working copy
  culib::transform3D(outputImage.getDevicePointer(), inputImage.getCudaArray(), outputImage.getSize(), scalingMatrix);

  boost::shared_ptr<tbblas::device_vector<double> > outputVB(new tbblas::device_vector<double>(outputCount));
  thrust::copy(thrust::device_ptr<float>(outputImage.getDevicePointer()),
      thrust::device_ptr<float>(outputImage.getDevicePointer()) + outputCount, outputVB->data().begin());
  output->setVisibleBiases(outputVB);

  //tbblas::device_vector<double>& inputHB = *input.getHiddenBiases();
  //boost::shared_ptr<tbblas::device_vector<double> > outputHB(new tbblas::device_vector<double>(inputHB.size()));
  //thrust::copy(inputHB.data().begin(), inputHB.data().end(), outputHB.data().begin());
  //output->setHiddenBiases(outputHB);

  // Scale the input weights matrix
  for (int i = 0; i < featureCount; ++i) {

  }

  output->setVisibleMean(input.getVisibleMean());
  output->setVisibleStd(input.getVisibleStd());
  output->setIsGaussian(input.getIsGaussian());

  data->setOutputModel(output);
}

}

}



