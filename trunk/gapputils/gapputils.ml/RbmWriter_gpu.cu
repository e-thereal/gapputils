/*
 * RbmWriter_gpu.cu
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include "RbmWriter.h"

#include <capputils/Verifier.h>
#include <thrust/copy.h>

#include <cstdio>
#include <iostream>

#include "RbmReader.h"

#define RBM_WRITER_ASSERT(pred) \
  if (!pred) { \
    std::cout << "[Error] Couldn't write file " << getFilename() << std::endl; \
    std::cout << "        Writing RBM aborted." << std::endl; \
    fclose(file); \
    return; \
  }

namespace gapputils {

namespace ml {

struct adiff : public thrust::binary_function<float, float, float> {
__host__ __device__
float operator()(const float& x, const float& y) {
  return fabs(x - y);
}

};

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

  thrust::copy(rbm.getVisibleMeans()->data().begin(),
      rbm.getVisibleMeans()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);

  thrust::copy(rbm.getVisibleStds()->data().begin(),
      rbm.getVisibleStds()->data().end(),
      buffer.begin());
  RBM_WRITER_ASSERT(fwrite(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);

  fclose(file);

  RbmReader reader;
  reader.setFilename(getFilename());
  reader.execute(0);
  reader.writeResults();

  RbmModel& rbm2 = *reader.getRbmModel();

  assert(rbm.getVisibleBiases()->size() == rbm2.getVisibleBiases()->size());
  assert(thrust::inner_product(rbm.getVisibleBiases()->data().begin(), rbm.getVisibleBiases()->data().end(),
      rbm2.getVisibleBiases()->data().begin(), 0.f, thrust::plus<float>(), adiff()) == 0);

  assert(rbm.getHiddenBiases()->size() == rbm2.getHiddenBiases()->size());
  assert(thrust::inner_product(rbm.getHiddenBiases()->data().begin(), rbm.getHiddenBiases()->data().end(),
      rbm2.getHiddenBiases()->data().begin(), 0.f, thrust::plus<float>(), adiff()) == 0);

  assert(rbm.getWeightMatrix()->data().size() == rbm2.getWeightMatrix()->data().size());
  assert(thrust::inner_product(rbm.getWeightMatrix()->data().begin(), rbm.getWeightMatrix()->data().end(),
      rbm2.getWeightMatrix()->data().begin(), 0.f, thrust::plus<float>(), adiff()) == 0);

  assert(rbm.getVisibleMeans()->size() == rbm2.getVisibleMeans()->size());
  assert(thrust::inner_product(rbm.getVisibleMeans()->begin(), rbm.getVisibleMeans()->end(),
      rbm2.getVisibleMeans()->begin(), 0.f, thrust::plus<float>(), adiff()) == 0);

  assert(rbm.getVisibleStds()->size() == rbm2.getVisibleStds()->size());
  assert(thrust::inner_product(rbm.getVisibleStds()->begin(), rbm.getVisibleStds()->end(),
      rbm2.getVisibleStds()->begin(), 0.f, thrust::plus<float>(), adiff()) == 0);
}

}

}


