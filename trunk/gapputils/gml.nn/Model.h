/*
 * Model.h
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef GML_MODEL_H_
#define GML_MODEL_H_

#include <tbblas/deeplearn/nn_model.hpp>
#include <tbblas/deeplearn/nn_patch_model.hpp>

namespace gml {

namespace nn {

typedef tbblas::deeplearn::nn_model<double> model_t;
typedef tbblas::deeplearn::nn_patch_model<double, 4> patch_model_t;

}

}

#endif /* GML_MODEL_H_ */
