/*
 * Model.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef GML_CONVRBM4D_MODEL_H_
#define GML_CONVRBM4D_MODEL_H_

#include <tbblas/deeplearn/conv_rbm_model.hpp>

namespace gml {

namespace convrbm4d {

typedef tbblas::deeplearn::conv_rbm_model<float, 4> model_t;

}

} /* namespace gml */

#endif /* GML_CONVRBM4D_MODEL_H_ */
