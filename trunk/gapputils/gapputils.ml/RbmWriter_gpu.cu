/*
 * RbmWriter_gpu.cu
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "RbmWriter.h"

#include <capputils/Verifier.h>
#include <thrust/copy.h>

#include <cstdio>
#include <iostream>

#define RBM_WRITER_ASSERT(pred) \
  if (!pred) { \
    std::cout << "[Error] Couldn't write file " << getFilename() << std::endl; \
    std::cout << "        Writing RBM aborted." << std::endl; \
    fclose(file); \
    return; \
  }

namespace gapputils {

namespace ml {

void RbmWriter::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmWriter();

  if (!capputils::Verifier::Valid(*this) || !getRbmModel())
    return;

  RbmModel& rbm = *getRbmModel();

  const int visibleCount = rbm.getVisibleBiases()->size();
  const int hiddenCount = rbm.getHiddenBiases()->size();

  FILE* file = fopen(getFilename().c_str(), "wb");
  if (!file) {
    std::cout << "[Error] Couldn't create file " << getFilename() << std::endl;
    std::cout << "        Writing RBM aborted." << std::endl;
    return;
  }

  std::vector<float> buffer(visibleCount * hiddenCount);

  // Start writing file
  RBM_WRITER_ASSERT(fwrite(&visibleCount, sizeof(int), 1, file) == 1);
  RBM_WRITER_ASSERT(fwrite(&hiddenCount, sizeof(int), 1, file) == 1);

  thrust::copy(rbm.getVisibleBiases()->data().begin(),
      rbm.getVisibleBiases()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);

  thrust::copy(rbm.getHiddenBiases()->data().begin(),
      rbm.getHiddenBiases()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), hiddenCount, file) == hiddenCount);

  thrust::copy(rbm.getWeightMatrix()->data().begin(),
      rbm.getWeightMatrix()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount * hiddenCount, file) == visibleCount * hiddenCount);

  thrust::copy(rbm.getVisibleStds()->data().begin(),
      rbm.getVisibleStds()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);

  thrust::copy(rbm.getVisibleMeans()->data().begin(),
      rbm.getVisibleMeans()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);

  fclose(file);
}

}

}


