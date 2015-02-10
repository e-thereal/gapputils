/*
 * AugmentDataset.cpp
 *
 *  Created on: Jan 23, 2015
 *      Author: tombr
 */

#include "AugmentDataset.h"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <tbblas/math.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(AugmentDataset)

  ReflectableBase(DefaultWorkflowElement<AugmentDataset>)

  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(ContrastSd)
  WorkflowProperty(BrightnessSd)
  WorkflowProperty(GammaSd)
  WorkflowProperty(SampleCount, Description("Number of samples generated per input tensor."))
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

AugmentDataset::AugmentDataset() : _ContrastSd(0), _BrightnessSd(0), _GammaSd(0), _SampleCount(10) {
  setLabel("Augment");
}

void AugmentDataset::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const int dimCount = host_tensor_t::dimCount;
  typedef host_tensor_t::dim_t dim_t;

  v_host_tensor_t& inputs = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t(inputs.size() * _SampleCount));

  boost::mt19937 rng; // I don't seed it on purpose (it's not relevant)
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

  host_tensor_t channel, output;
  for (size_t iInput = 0; iInput < inputs.size(); ++iInput) {
    host_tensor_t& input = *inputs[iInput];
    output.resize(input.size());

    dim_t channelSize = input.size();
    channelSize[dimCount - 1] = 1;

    dim_t topleft = seq<dimCount>(0);
    topleft[dimCount - 1] = 1;

    for (int iSample = 0; iSample < _SampleCount && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
      for (int iChannel = 0; iChannel < input.size()[dimCount - 1]; ++iChannel) {

        channel = input[topleft * iChannel, channelSize];

        const float slope = exp(var_nor() * _ContrastSd);
        const float gamma = exp(var_nor() * _GammaSd);
        const float intercept = var_nor() * _BrightnessSd;

//        dlog(Severity::Trace) << "Slope: " << slope << ", gamma: " << gamma << ", intercept: " << intercept;

        const float minValue = min(channel);
        const float maxValue = max(channel);

        // Standardize
        channel = (channel - minValue) / (maxValue - minValue);

        // Calculate new contrast
        channel = (slope * pow(channel, gamma) - 0.5f) + 0.5f + intercept;

        // Diversify
        channel = channel * (maxValue - minValue) + minValue;

        output[topleft * iChannel, channelSize] = channel;
      }

      outputs->at(iInput + iSample * inputs.size()) = boost::make_shared<host_tensor_t>(output);
    }
    if (monitor)
      monitor->reportProgress(100.0 * (iInput + 1) / inputs.size());
  }

  newState->setOutputs(outputs);
}

} /* namespace imageprocessing */

} /* namespace gml */
