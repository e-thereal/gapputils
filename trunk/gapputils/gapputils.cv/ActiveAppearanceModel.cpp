/*
 * ActiveAppearanceModel.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "ActiveAppearanceModel.h"

#include <capputils/VolatileAttribute.h>
#include <capputils/EnumerableAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ActiveAppearanceModel)

  DefineProperty(MeanGrid, Enumerable<boost::shared_ptr<std::vector<float> >, false>())
  DefineProperty(MeanImage, Enumerable<boost::shared_ptr<std::vector<float> >, false>())
  DefineProperty(PrincipalGrids, Enumerable<boost::shared_ptr<std::vector<float> >, false>())
  DefineProperty(PrincipalImages, Enumerable<boost::shared_ptr<std::vector<float> >, false>())
  DefineProperty(PrincipalParameters, Enumerable<boost::shared_ptr<std::vector<float> >, false>())
  DefineProperty(RowCount)
  DefineProperty(ColumnCount)
  DefineProperty(Width)
  DefineProperty(Height)

EndPropertyDefinitions

ActiveAppearanceModel::ActiveAppearanceModel()
 : _MeanGrid(new std::vector<float>()),
   _MeanImage(new std::vector<float>()),
   _PrincipalGrids(new std::vector<float>()),
   _PrincipalImages(new std::vector<float>()),
   _PrincipalParameters(new std::vector<float>())
{
}

ActiveAppearanceModel::~ActiveAppearanceModel() {
}

}

}
