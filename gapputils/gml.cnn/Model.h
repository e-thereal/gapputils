/*
 * Model.h
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef GML_MODEL_H_
#define GML_MODEL_H_

#include <tbblas/deeplearn/cnn_model.hpp>

namespace gml {

namespace cnn {

typedef tbblas::deeplearn::cnn_layer_model<float, 4> cnn_layer_t;
typedef tbblas::deeplearn::nn_layer_model<float> nn_layer_t;
typedef tbblas::deeplearn::cnn_model<float, 4> model_t;

}

}

#endif /* GML_MODEL_H_ */
