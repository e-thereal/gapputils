/*
 * ImagesInterface.h
 *
 *  Created on: Jul 26, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_IMAGESINTERFACE_H_
#define GAPPUTILS_IMAGESINTERFACE_H_

#include <gapputils/CollectionElement.h>

#include <gapputils/Image.h>

namespace interfaces {

namespace inputs {

class Images : public gapputils::workflow::CollectionElement {

  InitReflectableClass(Images)

  Property(Values, boost::shared_ptr<std::vector<boost::shared_ptr<gapputils::image_t> > >)
  Property(Value, boost::shared_ptr<gapputils::image_t>)

public:
  Images();
};

}

namespace outputs {

class Images : public gapputils::workflow::CollectionElement {

  InitReflectableClass(Images)

  Property(Values, boost::shared_ptr<std::vector<boost::shared_ptr<gapputils::image_t> > >)
  Property(Value, boost::shared_ptr<gapputils::image_t>)

public:
  Images();
};

}

} /* namespace interfaces */
#endif /* GAPPUTILS_IMAGESINTERFACE_H_ */
