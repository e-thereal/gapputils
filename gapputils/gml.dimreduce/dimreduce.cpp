/*
 * dimreduce.cpp
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#include "dimreduce.h"

#include <limits>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "dimred/ya_dimred.h"
using namespace yala;

#include <algorithm>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <capputils/Serializer.h>

namespace gml {

namespace dimreduce {

void trainModel(std::vector<boost::shared_ptr<std::vector<double> > >& input,
    DimensionalityReductionMethod method, int lowdim, int neighbors, Model& model)
{
  assert(input.size());

  const size_t rows = input.size(), cols = input[0]->size();

  std::vector<double> idata(rows * cols);
  std::vector<double> odata(rows * lowdim);

  for (size_t i = 0; i < input.size(); ++i) {
    std::copy(input[i]->begin(), input[i]->end(), idata.begin() + cols * i);
  }

  YA_WRAP_RM(double) input_w(&idata[0], rows, cols);
  YA_WRAP_RM(double) output_w(&odata[0], rows, lowdim);

  boost::shared_ptr<YADimReduce<double> > reductionMethod;

  switch(method) {
  case DimensionalityReductionMethod::PCA:
    reductionMethod = boost::make_shared<YAPCAReduce<double> >();
    break;

  case DimensionalityReductionMethod::LLE:
    reductionMethod = boost::make_shared<YALLEReduce<double> >();
    reductionMethod->neighbor_weight_mode(1);
    reductionMethod->neighbors(neighbors);
    reductionMethod->neighbor_mode(0);
    break;

  case DimensionalityReductionMethod::Isomap:
    reductionMethod = boost::make_shared<YAIsoReduce<double> >();
    reductionMethod->neighbor_weight_mode(0);
    reductionMethod->neighbors(neighbors);
    reductionMethod->neighbor_mode(0);
    break;
  }

  EigenOptions eigopts;
  reductionMethod->verbose(2);
  reductionMethod->find_t(input_w, output_w, lowdim, eigopts);

  model.setMethod(method);
  model.setModel(reductionMethod);
}

void encode(const std::vector<boost::shared_ptr<std::vector<double> > >& input,
    const Model& model, std::vector<boost::shared_ptr<std::vector<double> > >& output)
{
  assert(input.size());

  const size_t rows = input.size(), cols = input[0]->size();

  assert(cols == (size_t)model.getModel()->high_dim());

  const int lowdim = model.getModel()->low_dim();

  std::vector<double> idata(rows * cols);
  std::vector<double> odata(rows * lowdim);

  for (size_t i = 0; i < input.size(); ++i) {
    std::copy(input[i]->begin(), input[i]->end(), idata.begin() + cols * i);
  }

  YA_WRAP_RM(double) input_w(&idata[0], rows, cols);
  YA_WRAP_RM(double) output_w(&odata[0], rows, lowdim);

  model.getModel()->forward_t(input_w, output_w);

  output.clear();
  for (size_t i = 0; i < rows; ++i) {
    output.push_back(boost::make_shared<std::vector<double> >(odata.begin() + i * lowdim, odata.begin() + (i+1) * lowdim));
  }
}

void decode(const std::vector<boost::shared_ptr<std::vector<double> > >& input,
    const Model& model, std::vector<boost::shared_ptr<std::vector<double> > >& output)
{
  assert(input.size());

  const size_t rows = input.size(), cols = input[0]->size();

  assert(cols == (size_t)model.getModel()->low_dim());

  const int highdim = model.getModel()->high_dim();

  std::vector<double> idata(rows * cols);
  std::vector<double> odata(rows * highdim);

  for (size_t i = 0; i < input.size(); ++i) {
    std::copy(input[i]->begin(), input[i]->end(), idata.begin() + cols * i);
  }

  YA_WRAP_RM(double) input_w(&idata[0], rows, cols);
  YA_WRAP_RM(double) output_w(&odata[0], rows, highdim);

  model.getModel()->reverse_t(input_w, output_w);

  output.clear();
  for (size_t i = 0; i < rows; ++i) {
    output.push_back(boost::make_shared<std::vector<double> >(odata.begin() + i * highdim, odata.begin() + (i+1) * highdim));
  }
}

void save(const Model& model, std::ostream& out) {
  capputils::Serializer::WriteToFile(model, model.findProperty("Method"), out);
  model.getModel()->save_map(out);
}

void load(Model& model, std::istream& in) {
  capputils::Serializer::ReadFromFile(model, model.findProperty("Method"), in);
  switch (model.getMethod()) {
  case DimensionalityReductionMethod::PCA:
    model.setModel(boost::make_shared<YAPCAReduce<double> >());
    break;

  case DimensionalityReductionMethod::LLE:
    model.setModel(boost::make_shared<YALLEReduce<double> >());
    break;

  case DimensionalityReductionMethod::Isomap:
    model.setModel(boost::make_shared<YAIsoReduce<double> >());
    break;

  default:
    assert(false);
  }

  model.getModel()->load_map(in);
}

}

}
