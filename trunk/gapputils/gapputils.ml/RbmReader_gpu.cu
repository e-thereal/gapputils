/*
 * RbmReader_gpu.cu
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */
#define BOOST_TYPEOF_COMPLIANT
#include "RbmReader.h"

#include <capputils/Verifier.h>
#include <thrust/copy.h>

#include <cstdio>
#include <iostream>

#define RBM_READER_ASSERT(pred) \
  if (!pred) { \
    std::cout << "[Error] Couldn't read file " << getFilename() << std::endl; \
    std::cout << "        Reading RBM aborted." << std::endl; \
    fclose(file); \
    return; \
  }

namespace ublas = boost::numeric::ublas;

namespace gapputils {

namespace ml {

void RbmReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<RbmModel> rbm(new RbmModel());
  int visibleCount = 0, hiddenCount = 0;

  FILE* file = fopen(getFilename().c_str(), "rb");
  if (!file) {
    std::cout << "[Error] Couldn't open file " << getFilename() << std::endl;
    std::cout << "        Reading RBM aborted." << std::endl;
    fclose(file);
    return;
  }

  RBM_READER_ASSERT(fread(&visibleCount, sizeof(int), 1, file) == 1);
  RBM_READER_ASSERT(fread(&hiddenCount, sizeof(int), 1, file) == 1);

  const int count = visibleCount * hiddenCount;
  std::vector<float> buffer(count);
  boost::shared_ptr<tbblas::device_vector<float> > visibleBiases(new tbblas::device_vector<float>(visibleCount));
  boost::shared_ptr<tbblas::device_vector<float> > hiddenBiases(new tbblas::device_vector<float>(hiddenCount));
  boost::shared_ptr<tbblas::device_matrix<float> > weightMatrix(new tbblas::device_matrix<float>(visibleCount, hiddenCount));
  boost::shared_ptr<ublas::vector<float> > visibleMeans(new ublas::vector<float>(visibleCount));
  boost::shared_ptr<ublas::vector<float> > visibleStds(new ublas::vector<float>(visibleCount));

  RBM_READER_ASSERT(fread(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);
  thrust::copy(buffer.begin(), buffer.begin() + visibleCount, visibleBiases->data().begin());
  rbm->setVisibleBiases(visibleBiases);

  RBM_READER_ASSERT(fread(&buffer[0], sizeof(float), hiddenCount, file) == hiddenCount);
  thrust::copy(buffer.begin(), buffer.begin() + hiddenCount, hiddenBiases->data().begin());
  rbm->setHiddenBiases(hiddenBiases);

  RBM_READER_ASSERT(fread(&buffer[0], sizeof(float), count, file) == count);
  thrust::copy(buffer.begin(), buffer.begin() + count, weightMatrix->data().begin());
  rbm->setWeightMatrix(weightMatrix);

  RBM_READER_ASSERT(fread(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);
  thrust::copy(buffer.begin(), buffer.begin() + visibleCount, visibleMeans->begin());
  rbm->setVisibleMeans(visibleMeans);

  RBM_READER_ASSERT(fread(&buffer[0], sizeof(float), visibleCount, file) == visibleCount);
  thrust::copy(buffer.begin(), buffer.begin() + visibleCount, visibleStds->begin());
  rbm->setVisibleStds(visibleStds);

  fclose(file);

  data->setRbmModel(rbm);
  data->setVisibleCount(visibleCount);
  data->setHiddenCount(hiddenCount);
}

}

}
