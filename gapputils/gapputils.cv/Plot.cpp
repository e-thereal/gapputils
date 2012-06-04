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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Plot)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(X, Input(), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Y, Input(), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Format, Observe(PROPERTY_ID))
  DefineProperty(Plot, Output(""), Observe(PROPERTY_ID))

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

  //std::cout << "x size: " << get

  if (!getX() || !getY()) {
    std::cout << "[Warning] No input given. Abording!" << std::endl;
    return;
  }

  std::vector<float>& x = *getX();
  std::vector<float>& y = *getY();

  if (x.size() != y.size()) {
    std::cout << "[Warning] Size mismatch. " << x.size() << " != " << y.size() << std::endl;
    return;
  }

  const unsigned count = x.size();

  std::stringstream plot;

  plot << "\\documentclass{article}" << std::endl;
  plot << "\\usepackage{tikz}" << std::endl;
  plot << "\\usepackage{pgfplots}" << std::endl;
  plot << "\\begin{document}" << std::endl;
  plot << "\\begin{tikzpicture}" << std::endl;
  plot << "\\begin{axis}" << std::endl;
  if (getFormat().size())
    plot << "\\addplot[" << getFormat() << "] coordinates {" << std::endl;
  else
    plot << "\\addplot coordinates {" << std::endl;

  for (unsigned i = 0; i < count; ++i) {
    plot << "  (" << x[i] << ", " << y[i] << ")" << std::endl;
  }

  plot << "};" << std::endl;
  plot << "\\end{axis}" << std::endl;
  plot << "\\end{tikzpicture}" << std::endl;
  plot << "\\end{document}" << std::endl;

  data->setPlot(plot.str());
}

void Plot::writeResults() {
  if (!data)
    return;

  setPlot(data->getPlot());
}

}

}
