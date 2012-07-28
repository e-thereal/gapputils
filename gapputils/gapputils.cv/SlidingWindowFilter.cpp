/*
 * SlidingWindowFilter.cpp
 *
 *  Created on: May 30, 2012
 *      Author: tombr
 */

#include "SlidingWindowFilter.h"

#include <capputils/EnumeratorAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <deque>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(SlidingWindowFilter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Filter, Enumerator<AggregatorFunction>(), Observe(Id))
  DefineProperty(FilterSize, Observe(Id))
  DefineProperty(OutputImage, Output(""), Volatile(), ReadOnly(), Observe(Id))

EndPropertyDefinitions

SlidingWindowFilter::SlidingWindowFilter() : _FilterSize(5), data(0) {
  WfeUpdateTimestamp
  setLabel("SWF");

  Changed.connect(capputils::EventHandler<SlidingWindowFilter>(this, &SlidingWindowFilter::changedHandler));
}

SlidingWindowFilter::~SlidingWindowFilter() {
  if (data)
    delete data;
}

void SlidingWindowFilter::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void slidingMinimum(float* inbuf, int width, int height, int depth, float* outbuf, int K, gapputils::workflow::IProgressMonitor* monitor) {
  for (int z = 0; z < depth && (monitor ? !monitor->getAbortRequested() : true); ++z) {
    for (int y = 0; y < height && (monitor ? !monitor->getAbortRequested() : true); ++y) {

      const int offset = (z * height + y) * width;
      std::deque< std::pair<float, int> > window;

      // Three parts: pre-loading, sliding, unloading
      for (int k = 0, i = 0; k < K / 2; ++k, ++i) {
        while (!window.empty() && window.back().first >= inbuf[i + offset])
          window.pop_back();
        window.push_back(std::make_pair(inbuf[i + offset], i));
      }

      for (int x = 0, i = K / 2; x < width - K / 2; ++x, ++i) {
        while (!window.empty() && window.back().first >= inbuf[i + offset])
          window.pop_back();
        window.push_back(std::make_pair(inbuf[i + offset], i));

        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[x + offset] = window.front().first;
        //outbuf[x + offset] = inbuf[x + offset];
      }

      for (int x = width - K / 2, i = width; x < width; ++x, ++i) {
        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[x + offset] = window.front().first;
      }

      if (monitor)
        monitor->reportProgress(100. * (z * height + y) / (height * depth) / 2);
    }
  }

  for (int z = 0; z < depth && (monitor ? !monitor->getAbortRequested() : true); ++z) {
    for (int x = 0; x < width && (monitor ? !monitor->getAbortRequested() : true); ++x) {

      const int offset = z * height + x;
      const int pitch = width;
      std::deque< std::pair<float, int> > window;

      // Three parts: pre-loading, sliding, unloading
      for (int k = 0, i = 0; k < K / 2; ++k, ++i) {
        while (!window.empty() && window.back().first >= outbuf[i * pitch + offset])
          window.pop_back();
        window.push_back(std::make_pair(outbuf[i * pitch + offset], i));
      }

      for (int y = 0, i = K / 2; y < height - K / 2; ++y, ++i) {
        while (!window.empty() && window.back().first >= outbuf[i * pitch + offset])
          window.pop_back();
        window.push_back(std::make_pair(outbuf[i * pitch + offset], i));

        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[y * pitch + offset] = window.front().first;
      }

      for (int y = height - K / 2, i = height; y < height; ++y, ++i) {
        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[y * pitch + offset] = window.front().first;
      }

      if (monitor)
        monitor->reportProgress(100. * (z * height + x) / (width * depth) / 2 + 50);
    }
  }
}

void slidingMaximum(float* inbuf, int width, int height, int depth, float* outbuf, int K, gapputils::workflow::IProgressMonitor* monitor) {
  for (int z = 0; z < depth && (monitor ? !monitor->getAbortRequested() : true); ++z) {
    for (int y = 0; y < height && (monitor ? !monitor->getAbortRequested() : true); ++y) {

      const int offset = (z * height + y) * width;
      std::deque< std::pair<float, int> > window;

      // Three parts: pre-loading, sliding, unloading
      for (int k = 0, i = 0; k < K / 2; ++k, ++i) {
        while (!window.empty() && window.back().first <= inbuf[i + offset])
          window.pop_back();
        window.push_back(std::make_pair(inbuf[i + offset], i));
      }

      for (int x = 0, i = K / 2; x < width - K / 2; ++x, ++i) {
        while (!window.empty() && window.back().first <= inbuf[i + offset])
          window.pop_back();
        window.push_back(std::make_pair(inbuf[i + offset], i));

        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[x + offset] = window.front().first;
      }

      for (int x = width - K / 2, i = width; x < width; ++x, ++i) {
        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[x + offset] = window.front().first;
      }

      if (monitor)
        monitor->reportProgress(100. * (z * height + y) / (height * depth) / 2);
    }
  }

  for (int z = 0; z < depth && (monitor ? !monitor->getAbortRequested() : true); ++z) {
    for (int x = 0; x < width && (monitor ? !monitor->getAbortRequested() : true); ++x) {

      const int offset = z * height + x;
      const int pitch = width;
      std::deque< std::pair<float, int> > window;

      // Three parts: pre-loading, sliding, unloading
      for (int k = 0, i = 0; k < K / 2; ++k, ++i) {
        while (!window.empty() && window.back().first <= outbuf[i * pitch + offset])
          window.pop_back();
        window.push_back(std::make_pair(outbuf[i * pitch + offset], i));
      }

      for (int y = 0, i = K / 2; y < height - K / 2; ++y, ++i) {
        while (!window.empty() && window.back().first <= outbuf[i * pitch + offset])
          window.pop_back();
        window.push_back(std::make_pair(outbuf[i * pitch + offset], i));

        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[y * pitch + offset] = window.front().first;
      }

      for (int y = height - K / 2, i = height; y < height; ++y, ++i) {
        while(window.front().second <= i - K)
          window.pop_front();

        outbuf[y * pitch + offset] = window.front().first;
      }

      if (monitor)
        monitor->reportProgress(100. * (z * height + x) / (width * depth) / 2 + 50);
    }
  }
}

void SlidingWindowFilter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new SlidingWindowFilter();

  if (!capputils::Verifier::Valid(*this) || !getInputImage())
    return;

  image_t& input = *getInputImage();
  const int width = input.getSize()[0];
  const int height = input.getSize()[1];
  const int depth = input.getSize()[2];

  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));
  float* inbuf = input.getData();
  float* outbuf = output->getData();
  const int K = getFilterSize();
  
  // Calculate sliding window over x axis
  // pair<int, int> represents the pair (ARR[i], i)
  
  switch (getFilter()) {
  case AggregatorFunction::Minimum:
    slidingMinimum(inbuf, width, height, depth, outbuf, K, monitor);
    break;

  case AggregatorFunction::Maximum:
    slidingMaximum(inbuf, width, height, depth, outbuf, K, monitor);
    break;
  }
  
  data->setOutputImage(output);
}

void SlidingWindowFilter::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
