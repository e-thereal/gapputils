/*
 * Image.h
 *
 *  Created on: Jul 16, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_IMAGE_H_
#define GAPPUTILS_IMAGE_H_

namespace gapputils {

template<class T, unsigned dim>
class ImageBase {
public:
  const static unsigned dimCount = dim;

  typedef unsigned size_t;
  typedef T value_t;
  typedef size_t dim_t[dim];

  typedef value_t* iterator;

protected:
  dim_t size;
  dim_t pixelSize;   ///< In microns
  value_t* data;

public:
  ImageBase(const dim_t& size, const dim_t& pixelSize) {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      this->size[i] = size[i];
      this->pixelSize[i] = pixelSize[i];
      count *= size[i];
    }
    data = new value_t[count];
  }

protected:
  ImageBase(const size_t& count) : data(new value_t[count]) { }

  ImageBase(const size_t& count, const dim_t& pixelSize) : data(new value_t[count]) {
    for (unsigned i = 0; i < dimCount; ++i) {
      this->pixelSize[i] = pixelSize[i];
    }
  }

public:

  virtual ~ImageBase() {
    delete data;
  }

  const dim_t& getSize() const { return size; }
  const dim_t& getPixelSize() const { return pixelSize; }
  value_t* getData() const { return data; }

  size_t getCount() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      count *= size[i];
    }
    return count;
  }

  void fill(const value_t& value) {
    size_t count = getCount();
    for (size_t i = 0; i < count; ++i)
      data[i] = value;
  }

  iterator begin() {
    return data;
  }

  iterator end() {
    return data + getCount();
  }
};

template<class T, unsigned dim>
class Image : public ImageBase<T, dim> { };

template<class T>
class Image3d : public ImageBase<T, 3u> {
public:
  typedef ImageBase<T, 3u> Base;

  const static unsigned dimCount = Base::dimCount;

  typedef typename Base::size_t size_t;
  typedef typename Base::value_t value_t;
  typedef typename Base::dim_t dim_t;

public:
  Image3d(const dim_t& size, const dim_t& pixelSize) : Base(size, pixelSize) { }

  Image3d(size_t width, size_t height, size_t depth,
      size_t pixelWidth = 1000, size_t pixelHeight = 1000, size_t pixelDepth = 1000)
   : Base(width * height * depth)
  {
    this->size[0] = width;
    this->size[1] = height;
    this->size[2] = depth;
    this->pixelSize[0] = pixelWidth;
    this->pixelSize[1] = pixelHeight;
    this->pixelSize[2] = pixelDepth;
  }

  Image3d(size_t width, size_t height, size_t depth,
      const dim_t& pixelSize)
   : Base(width * height * depth, pixelSize)
  {
    this->size[0] = width;
    this->size[1] = height;
    this->size[2] = depth;
  }
};

typedef Image3d<float> image_t;

} /* namespace gapputils */
#endif /* GAPPUTILS_IMAGE_H_ */
