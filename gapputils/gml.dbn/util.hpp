/*
 * util.hpp
 *
 *  Created on: Apr 7, 2014
 *      Author: tombr
 */

#ifndef GML_DBN_UTIL_HPP_
#define GML_DBN_UTIL_HPP_

namespace gml {

namespace dbn {

void enable_peer_access(int gpu_count = 1);
void disable_peer_access(int gpu_count = 1);

}

}


#endif /* GML_DBN_UTIL_HPP_ */
