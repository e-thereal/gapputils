/*
 * Checkerboard.cpp
 *
 *  Created on: Jan 25, 2012
 *      Author: tombr
 */

#include "Checkerboard.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Checkerboard)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Width, Observe(PROPERTY_ID))
  DefineProperty(Height, Observe(PROPERTY_ID))
  DefineProperty(Depth, Observe(PROPERTY_ID))
  DefineProperty(TileWidth, Observe(PROPERTY_ID))
  DefineProperty(TileHeight, Observe(PROPERTY_ID))
  DefineProperty(TileDepth, Observe(PROPERTY_ID))
  DefineProperty(DarkValue, Observe(PROPERTY_ID))
  DefineProperty(LightValue, Observe(PROPERTY_ID))
  DefineProperty(Checkerboard, Output("Board"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

Checkerboard::Checkerboard()
 : _Width(1), _Height(1), _Depth(1), _TileWidth(1), _TileHeight(1), _TileDepth(1),
   _DarkValue(0), _LightValue(1), data(0)
{
  WfeUpdateTimestamp
  setLabel("Checkerboard");

  Changed.connect(capputils::EventHandler<Checkerboard>(this, &Checkerboard::changedHandler));
}

Checkerboard::~Checkerboard() {
  if (data)
    delete data;
}

void Checkerboard::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Checkerboard::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Checkerboard();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int width = getWidth(), height = getHeight(), depth = getDepth(),
      w = getTileWidth(), h = getTileHeight(), d = getTileDepth();
  const double dark = getDarkValue(), light = getLightValue();

  boost::shared_ptr<std::vector<double> > board(new std::vector<double>(width * height * depth));

  for (int i = 0, z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {
        board->at(i) = ((x / w + y / h + z / d) % 2 ? dark : light);
      }
    }
  }

  data->setCheckerboard(board);
}

void Checkerboard::writeResults() {
  if (!data)
    return;

  setCheckerboard(data->getCheckerboard());
}

}

}
