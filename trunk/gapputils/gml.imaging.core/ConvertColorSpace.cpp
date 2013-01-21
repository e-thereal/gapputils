#include "ConvertColorSpace.h"

#include "math3d.h"

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(ConvertColorSpace)

  ReflectableBase(DefaultWorkflowElement<ConvertColorSpace>)

  WorkflowProperty(InputImage, Input("In"), NotNull<Type>())
  WorkflowProperty(InputColorSpace, Enumerator<Type>())
  WorkflowProperty(OutputColorSpace, Enumerator<Type>())
  WorkflowProperty(OutputImage, Output("Out"))

EndPropertyDefinitions

ConvertColorSpace::ConvertColorSpace() {
  setLabel("Rgb2Rgb");
}

void ConvertColorSpace::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  const int slicePitch = input.getSize()[0] * input.getSize()[1];
  float* inData = input.getData();
  float* outData = output->getData();

  const ColorSpace inSpace = getInputColorSpace();
  const ColorSpace outSpace = getOutputColorSpace();

  for (size_t idx = 0; idx < input.getCount() / 3; ++idx) {
    float X = 0, Y = 0, Z = 0;
    const size_t i = idx + 3 * (idx / slicePitch);

    switch (inSpace) {
    case ColorSpace::XYZ:
      X = inData[i + 3 * (i / slicePitch)];
      Y = inData[i + slicePitch];
      Z = inData[i + 2 * slicePitch];
      break;

    case ColorSpace::xyY:
      {
        const float x = inData[i];
        const float y = inData[i + slicePitch];
        Y = inData[i + 2 * slicePitch];
        X = Y / y * x;
        Z = Y / y * (1 - x - y);
      }
      break;

    case ColorSpace::sRGB:
      {
        const float R = inData[i];
        const float G = inData[i + slicePitch];
        const float B = inData[i + 2 * slicePitch];
        float4 rgb = make_float4(R <= 0.00313088f ? R / 12.92f : powf((R + 0.055f) / 1.055f, 2.4f),
          G <= 0.00313088f ? G / 12.92f : powf((G + 0.055f) / 1.055f, 2.4f),
          B <= 0.00313088f ? B / 12.92f : powf((B + 0.055f) / 1.055f, 2.4f),
          1);

        gml::fmatrix4 rgb2xyz = gml::make_fmatrix4(0.4124564f, 0.3575761f, 0.1804375f, 0,
                                                   0.2126729f, 0.7151522f, 0.0721750f, 0,
                                                   0.0193339f, 0.1191920f, 0.9503041f, 0,
                                                   0, 0, 0, 1);
        gml::float4 xyz = rgb2xyz * rgb;

        X = gml::get_x(xyz);
        Y = gml::get_y(xyz);
        Z = gml::get_z(xyz);
      }
      break;

    default:
      dlog(Severity::Warning) << "Input color space '" << inSpace << "' has not yet been implemented.";
      return;
    }

    switch (outSpace) {
    case ColorSpace::XYZ:
      outData[i] = X;
      outData[i + slicePitch] = Y;
      outData[i + 2 * slicePitch] = Z;
      break;

    case ColorSpace::xyY:
      outData[i] = (float)((double)X / ((double)X + Y + Z + 1e-7));
      outData[i + slicePitch] = (float)((double)Y / ((double)X + Y + Z + 1e-7));
      outData[i + 2 * slicePitch] = Y;
      break;

    case ColorSpace::sRGB:
      {
        gml::fmatrix4 xyz2rgb = gml::make_fmatrix4(3.2404542f, -1.5371385f, -0.4985314f, 0,
                                                  -0.9692660f,  1.8760108f,  0.0415560f, 0,
                                                    0.055644f, -0.2040259f,  1.0572252f, 0,
                                                    0, 0, 0, 1);
        gml::float4 xyz = make_float4(X, Y, Z, 1);
        gml::float4 rgb = xyz2rgb * xyz;

        const float r = gml::get_x(rgb);
        const float g = gml::get_y(rgb);
        const float b = gml::get_z(rgb);
        outData[i] = (float)(r <= 0.00313088 ? 12.92 * r : 1.055 * pow((double)r, 1.0 / 2.4) - 0.055);
        outData[i + slicePitch] = (float)(g <= 0.00313088 ? 12.92 * g : 1.055 * pow((double)g, 1.0 / 2.4) - 0.055);
        outData[i + 2 * slicePitch] = (float)(b <= 0.00313088 ? 12.92 * b : 1.055 * pow((double)b, 1.0 / 2.4) - 0.055);
      }
      break;

    default:
      dlog(Severity::Warning) << "Output color space '" << outSpace << "' has not yet been implemented.";
      return;
    }
  }

  newState->setOutputImage(output);
}

}

}

}
