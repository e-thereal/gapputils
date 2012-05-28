
#include "SimilarityNCC.h"

#include "ICudaImage.h"

using namespace culib;
using namespace reglib;
using namespace optlib;

namespace regutil {

SimilarityNCC::SimilarityNCC(void)
{
}

SimilarityNCC::~SimilarityNCC(void)
{
}

double SimilarityNCC::eval(const ImagePair<int3>& images) {
  const ICudaImage* img1 = 0;
  const ICudaImage* img2 = 0;
  if ((img1 = dynamic_cast<const ICudaImage*>(images.image1)) &&
      (img2 = dynamic_cast<const ICudaImage*>(images.image2)))
  {
    double value = calculateNCC(img1->getDevicePointer(),
        img2->getDevicePointer(), img1->getSize(), dim3(xskip, yskip, zskip));
    return value;
  }
  return 0.0;
}

void SimilarityNCC::setParameter(int id, double value) {
  switch(id) {
    case XSkip: xskip = value; break;
    case YSkip: yskip = value; break;
    case ZSkip: zskip = value; break;
  }
}

void SimilarityNCC::setParameter(int id, void* value) {
}

}
