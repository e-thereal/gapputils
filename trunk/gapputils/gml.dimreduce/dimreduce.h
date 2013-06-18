/*
 * dimreduce.h
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#ifndef GML_DIMREDUCE_H_
#define GML_DIMREDUCE_H_

#include "Model.h"
#include "DimensionalityReductionMethod.h"

#include <iostream>

namespace gml {

namespace dimreduce {

void trainModel(std::vector<boost::shared_ptr<std::vector<double> > >& input,
    DimensionalityReductionMethod method, int lowdim, int neighbors, Model& model);

void encode(const std::vector<boost::shared_ptr<std::vector<double> > >& input,
    const Model& model, std::vector<boost::shared_ptr<std::vector<double> > >& output);
void decode(const std::vector<boost::shared_ptr<std::vector<double> > >& input,
    const Model& model, std::vector<boost::shared_ptr<std::vector<double> > >& output);

void save(const Model& model, std::ostream& out);
void load(Model& model, std::istream& in);

}

}

#endif /* GML_DIMREDUCE_H_ */
