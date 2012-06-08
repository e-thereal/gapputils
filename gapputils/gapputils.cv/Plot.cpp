/*
 * Plot.cpp
 *
 *  Created on: May 20, 2012
 *      Author: tombr
 */

#include "Plot.h"

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

#include <culib/CudaImage.h>
#include <regutil/CudaImage.h>

#include <algorithm>
#include <sstream>
#include <iostream>

#include <culib/histogram.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Plot)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(OldPlot, Input("Plot"), Hide(), Observe(PROPERTY_ID))
  DefineProperty(X, Input(), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Y, Input(), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Image, Input("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Format, Observe(PROPERTY_ID))
  DefineProperty(Plot, Output(""), Hide(), Observe(PROPERTY_ID))

EndPropertyDefinitions

Plot::Plot() : data(0) {
  WfeUpdateTimestamp
  setLabel("Plot");

  Changed.connect(capputils::EventHandler<Plot>(this, &Plot::changedHandler));
}

Plot::~Plot() {
  if (data)
    delete data;
}

void Plot::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Plot::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Plot();

  if (!capputils::Verifier::Valid(*this))
    return;

  const std::string placeholder = "% new plot placeholder";

  //std::cout << "x size: " << get

  

  std::stringstream plot;

  if (getOldPlot().size()) {
    std::stringstream oldPlot(getOldPlot());
    std::string line;
    while (getline(oldPlot, line)) {
      if (line == placeholder) {
        if (getX() && getY()) {
          std::vector<float>& x = *getX();
          std::vector<float>& y = *getY();

          if (x.size() == y.size()) {
            const unsigned count = x.size();

            if (getFormat().size())
              plot << "\\addplot[" << getFormat() << "] coordinates {" << std::endl;
            else
              plot << "\\addplot coordinates {" << std::endl;

            for (unsigned i = 0; i < count; ++i) {
              plot << "  (" << x[i] << ", " << y[i] << ")" << std::endl;
            }
            plot << "};" << std::endl;
          } else {
            std::cout << "[Warning] Size mismatch. " << x.size() << " != " << y.size() << std::endl;
          }
        }

        if (getImage()) {
          culib::ICudaImage& image = *getImage();
          culib::HistogramConfig config;
          culib::setupHistogramConfig(config, dim3(256), make_float2(1.f / 256.f));
          getHistogram(config, image.getDevicePointer(), image.getSize());

          //thrust::device_ptr<uint> histogram(config.d_histogram);
          //thrust::device_vector<float> cdf(histogram, histogram + binCount);
  
          culib::cleanupHistogramConfig(config);
        }
      }
      plot << line << std::endl;
    }
  } else {
    plot << "\\documentclass[tikz,border=0.5cm]{standalone}" << std::endl;
    plot << "\\usepackage{pgfplots}" << std::endl;
    plot << "\\usepackage{cmbright}" << std::endl;
    plot << "\\tikzstyle{every node}+=[font=\\sffamily]" << std::endl;
    plot << "\\begin{document}" << std::endl;
    plot << "\\begin{tikzpicture}" << std::endl;
    plot << "\\begin{axis}[font=\\sffamily]" << std::endl;

    if (getX() && getY()) {
      std::vector<float>& x = *getX();
      std::vector<float>& y = *getY();

      if (x.size() == y.size()) {
        const unsigned count = x.size();

        if (getFormat().size())
          plot << "\\addplot[" << getFormat() << "] coordinates {" << std::endl;
        else
          plot << "\\addplot coordinates {" << std::endl;

        for (unsigned i = 0; i < count; ++i) {
          plot << "  (" << x[i] << ", " << y[i] << ")" << std::endl;
        }
        plot << "};" << std::endl;
      } else {
        std::cout << "[Warning] Size mismatch. " << x.size() << " != " << y.size() << std::endl;
      }
    }

    plot << placeholder << std::endl;
    plot << "\\end{axis}" << std::endl;
    plot << "\\end{tikzpicture}" << std::endl;
    plot << "\\end{document}" << std::endl;
  }

  data->setPlot(plot.str());
}

void Plot::writeResults() {
  if (!data)
    return;

  setPlot(data->getPlot());
}

}

}
