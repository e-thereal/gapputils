/*
 * SimilarityShannon.cpp
 *
 *  Created on: Nov 13, 2008
 *      Author: tombr
 */

#include "SimilarityShannon.h"
#include "ICudaImage.h"

#include <iostream>

using namespace culib;
using namespace optlib;
using namespace reglib;

using namespace std;

namespace regutil {

SimilarityShannon::SimilarityShannon(dim3 binCount, float2 binScale, SimilarityConfig::Method method) : xskip(1),
    yskip(1), zskip(1)
{
  setupSimilarityConfig(simConfig, binCount, binScale);
  simConfig.method = method;
}

SimilarityShannon::~SimilarityShannon() {
  cleanupSimilarityConfig(simConfig);
}

double SimilarityShannon::eval(const ImagePair<int3>& images) {
  const ICudaImage* img1 = 0;
  const ICudaImage* img2 = 0;
  if ((img1 = dynamic_cast<const ICudaImage*>(images.image1)) &&
      (img2 = dynamic_cast<const ICudaImage*>(images.image2)))
  {
    double value = getSimilarity(simConfig, img1->getDevicePointer(),
        img2->getDevicePointer(), img1->getSize(), dim3(xskip, yskip, zskip));
    return value;
  }
  cout << "No ICudaImage found" << endl;
  return 0.0;
}

void SimilarityShannon::setParameter(int id, double value) {
  switch(id) {
    case XSkip: xskip = value; break;
    case YSkip: yskip = value; break;
    case ZSkip: zskip = value; break;
    case HistogramSigma: simConfig.sigma = value; break;
    case SimilarityMode: simConfig.computationLevel = int(value); break;
    case SimilarityMethod: simConfig.method = int(value); break;
  }
}

void SimilarityShannon::setParameter(int id, void* value) {
}

}
