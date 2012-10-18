/*
 * Interfaces.h
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_CV_INTERFACES_H_
#define GAPPUTILS_CV_INTERFACES_H_

#include <qimage.h>

#include <capputils/ReflectableClass.h>

namespace gapputils {
namespace cv {

class Interfaces : capputils::reflection::ReflectableClass {
  InitReflectableClass(Interfaces)

  Property(Image, boost::shared_ptr<QImage>)

};

} /* namespace ml */
} /* namespace gapputils */
#endif /* GAPPUTILS_CV_INTERFACES_H_ */
