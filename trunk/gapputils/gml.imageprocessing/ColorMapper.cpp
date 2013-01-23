/*
 * ColorMapper.cpp
 *
 *  Created on: Jan 22, 2013
 *      Author: tombr
 */

#include "ColorMapper.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ColorMapper)

  ReflectableBase(DefaultWorkflowElement<ColorMapper>)

  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(ColorMap, Enumerator<Type>())
  WorkflowProperty(MinimumIntensity)
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

ColorMapper::ColorMapper() : _MinimumIntensity(0.0), _MaximumIntensity(1.0) {
  setLabel("Grey");
}

void getHeatMap1Color(float value, float *red, float *green, float *blue) {
  const int NUM_COLORS = 4;
  static float color[NUM_COLORS][3] = { {0,0,1}, {0,1,0}, {1,1,0}, {1,0,0} };
    // a static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each

  int idx1;        // |-- our desired color will be between these two indexes in "color"
  int idx2;        // |
  float fractBetween = 0;  // fraction between "idx1" and "idx2" where our value is

  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // will multiply value by 3
    idx1  = floor(value);                  // our desired color will be after this index
    idx2  = idx1+1;                        // ... and before this index (inclusive)
    fractBetween = value - float(idx1);    // distance between the two indexes (0-1)
  }

  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

void getHeatMap2Color(float value, float *red, float *green, float *blue) {
  const int NUM_COLORS = 6;
  static float color[NUM_COLORS][3] = {{0,0,0}, {0,0,0.5}, {0.5,0,0.5}, {1,0,0}, {1,1,0}, {1,1,1}};
    // a static array of 6 colors:  (black,      blue,   violet,  red,  yellow,     white) using {r,g,b} for each

  int idx1;        // |-- our desired color will be between these two indexes in "color"
  int idx2;        // |
  float fractBetween = 0;  // fraction between "idx1" and "idx2" where our value is

  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // will multiply value by 3
    idx1  = floor(value);                  // our desired color will be after this index
    idx2  = idx1+1;                        // ... and before this index (inclusive)
    fractBetween = value - float(idx1);    // distance between the two indexes (0-1)
  }

  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

void ColorMapper::update(IProgressMonitor* monitor) const {
  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize()[0], input.getSize()[1], 3 * input.getSize()[2], input.getPixelSize()));

  const int slicePitch = input.getSize()[0] * input.getSize()[1];
  float* inData = input.getData();
  float* outData = output->getData();
  ColorMap colorMap = getColorMap();

  float minimum = getMinimumIntensity();
  float range = getMaximumIntensity() - getMinimumIntensity();

  for (size_t idx = 0; idx < input.getCount(); ++idx) {
    float r = 0, g = 0, b = 0;
    const size_t i = idx + 2 * (idx / slicePitch) * slicePitch;
    float value = (inData[idx] - minimum) / range;

    switch (colorMap) {
    case ColorMap::Greyscale:
      r = g = b = value;
      break;

    case ColorMap::HeatMap1:
      getHeatMap1Color(value, &r, &g, &b);
      break;

    case ColorMap::HeatMap2:
      getHeatMap2Color(value, &r, &g, &b);
      break;
    }
    outData[i] = r;
    outData[i + slicePitch] = g;
    outData[i + 2 * slicePitch] = b;
  }

  newState->setOutputImage(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
