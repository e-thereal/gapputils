/*
 * Histogram.cpp
 *
 *  Created on: Oct 24, 2012
 *      Author: tombr
 */

#include "Histogram.h"

#include <cmath>
#include <set>
#include <map>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace cv {

BeginPropertyDefinitions(Histogram)

  ReflectableBase(workflow::DefaultWorkflowElement<Histogram>)

  WorkflowProperty(Image, Input("Img"), NotNull<Type>())
  WorkflowProperty(BinCount, NotEqual<Type>(0))
  WorkflowProperty(HistogramBinWidth, NotEqual<Type>(0))
  WorkflowProperty(HistogramHeight, NotEqual<Type>(0))
  WorkflowProperty(AverageHeight, NotEqual<Type>(0))
  WorkflowProperty(Foreground)
  WorkflowProperty(Background)
  WorkflowProperty(ModeColor)
  WorkflowProperty(ModeRadius, NotEqual<Type>(0))
  WorkflowProperty(MinMode)
  WorkflowProperty(ModeCount, Description("If -1, all modes are calculated."))
  WorkflowProperty(SmoothingRadius, Description("A value of 0 indicates no smoothing."))
  WorkflowProperty(Histogram, Output("Hist"))
  WorkflowProperty(Modes, Output("M"))

EndPropertyDefinitions

Histogram::Histogram() : _BinCount(256), _HistogramBinWidth(1), _HistogramHeight(256),
 _AverageHeight(128), _Foreground(0.5f), _Background(0.0f),
 _ModeColor(1.0f), _ModeRadius(1), _MinMode(0), _ModeCount(-1), _SmoothingRadius(0)
{
  setLabel("Histogram");
}

Histogram::~Histogram() {
}

void Histogram::update(workflow::IProgressMonitor* monitor) const {
  using namespace std;
  using namespace capputils;

  Logbook& dlog = getLogbook();

  image_t& image = *getImage();

  size_t binCount = getBinCount();
  size_t count = image.getCount();
  float* data = image.getData();

  // Populate histogram
  std::vector<int> bins(binCount);
  for (size_t i = 0; i < count; ++i) {
    const int idx = min<int>(binCount - 1, max<int>(0, data[i] * (binCount - 1)));
    ++bins[idx];
  }

  // Smooth histogram
  std::vector<float> smoothedBins(binCount);
  for (int i = 0; i < (int)binCount; ++i) {
    int sampleCount = 1;
    float bin = bins[i];
    for (int j = 1; j <= getSmoothingRadius(); ++j) {
      if (i - j >= 0) {
        ++sampleCount;
        bin += (float)bins[i - j];
      }
      if (i + j < (int)binCount) {
        ++sampleCount;
        bin += (float)bins[i + j];
      }
    }
    smoothedBins[i] = bin / (float)sampleCount;
  }

  // Find modes
  const int rMode = getModeRadius();
  std::set<int> modes;
  for (int i = rMode + getMinMode(); i < (int)binCount - rMode; ++i) {
    if (smoothedBins[i - rMode] <= smoothedBins[i] && smoothedBins[i + rMode] <= smoothedBins[i])
      modes.insert(i);
  }

  // Sharpen modes
  std::vector<int> modesVec;
  for (std::set<int>::iterator i = modes.begin(); i != modes.end(); ++i)
    modesVec.push_back(*i);
  modes.clear();

  if (modesVec.size() < 1) {
    dlog(Severity::Warning) << "No mode found.";
  } else {
    int sum = modesVec[0];
    int sampleCount = 1;
    for (size_t i = 1; i < modesVec.size(); ++i) {
      if (modesVec[i] - modesVec[i-1] > 1) {
        modes.insert(sum / sampleCount);
        sum = 0;
        sampleCount = 0;
      }
      sum += modesVec[i];
      ++sampleCount;
    }
    modes.insert(sum / sampleCount);
  }

  if (getModeCount() >= 0) {
    std::multimap<float, int> maxModes;
    for (std::set<int>::iterator i = modes.begin(); i != modes.end(); ++i) {
      float c = -smoothedBins[*i];
      int pos = *i;
      maxModes.insert(std::make_pair(c, pos));
    }

    modes.clear();

    for (int i = 0; i < 3 && maxModes.size(); ++i) {
      dlog(Severity::Trace) << i + 1 << ". modes: bin[" << maxModes.begin()->second << "] = " << -maxModes.begin()->first;
      modes.insert(maxModes.begin()->second);
      maxModes.erase(maxModes.begin());
    }
  }

  // Plot histogram
  size_t binWidth = getHistogramBinWidth();
  size_t width = binCount * binWidth;
  size_t height = getHistogramHeight();
  boost::shared_ptr<image_t> histogram(new image_t(width, height, 1));

  float* histData = histogram->getData();
  size_t histCount = histogram->getCount();
  size_t avgHeight = getAverageHeight();

  const float bg = getBackground();

  for (size_t i = 0; i < histCount; ++i) {
    const size_t idx = (i / binWidth) % binCount;
    const float fg = modes.find(idx) == modes.end() ? getForeground() : getModeColor();
    histData[i] = ((float)smoothedBins[idx] / (float)count * (float)binCount * (float)avgHeight > height - i / width ? fg : bg);
  }

  newState->setHistogram(histogram);

  // Write modes vector
  boost::shared_ptr<std::vector<double> > outModes(new std::vector<double>());

  for (std::set<int>::iterator i = modes.begin(); i != modes.end(); ++i) {
    outModes->push_back((double)*i / (double)(binCount - 1));
  }

  newState->setModes(outModes);
}

} /* namespace cv */

} /* namespace gapputils */
