/*
 * Model.h
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef GML_MODEL_H_
#define GML_MODEL_H_

#include <tbblas/deeplearn/encoder_model.hpp>

namespace gml {

namespace encoder {

typedef tbblas::deeplearn::cnn_layer_model<float, 4> cnn_layer_t;
typedef tbblas::deeplearn::dnn_layer_model<float, 4> dnn_layer_t;
typedef tbblas::deeplearn::nn_layer_model<float> nn_layer_t;
typedef tbblas::deeplearn::encoder_model<float, 4> model_t;

}

}

#endif /* GML_MODEL_H_ */
