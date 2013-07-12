/*
 * OldMifReader.cpp
 *
 *  Created on: Jul 5, 2013
 *      Author: tombr
 */

#include "OldMifReader.h"

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(OldMifReader)

  ReflectableBase(DefaultWorkflowElement<OldMifReader>)

  WorkflowProperty(MifName, Input("Mif"), Filename("MIFs (*.MIF *.MIF.gz)"), FileExists())
  WorkflowProperty(Image, Output("Img"))
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())

EndPropertyDefinitions

OldMifReader::OldMifReader() : _MaximumIntensity(2048), _Width(0), _Height(0), _Depth(0)
{
  setLabel("Mif");
}

void OldMifReader::update(IProgressMonitor* monitor) const {

}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
