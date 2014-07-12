/*
 * Model.h
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#ifndef GML_MODEL_H_
#define GML_MODEL_H_

#include <tbblas/deeplearn/dbn_model.hpp>

namespace gml {

namespace dbn {

typedef tbblas::deeplearn::dbn_model<float, 4> dbn_t;
typedef tbblas::deeplearn::conv_rbm_model<float, 4> crbm_t;
typedef tbblas::deeplearn::rbm_model<double> rbm_t;

}

}

#endif /* GML_MODEL_H_ */
