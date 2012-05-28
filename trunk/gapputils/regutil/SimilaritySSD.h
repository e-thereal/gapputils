#pragma once

#ifndef REGUTIL_SIMILARITYSSD_H_
#define REGUTIL_SIMILARITYSSD_H_

#include <reglib/RegistrationProblem.h>
#include <optlib/IParameterizable.h>

#include <culib/similarity.h>

namespace regutil {

class SimilaritySSD : public virtual reglib::ISimilarityMeasure<int3>,
                      public virtual optlib::IParameterizable
{
private:
  int xskip,                            ///< Every xskip voxel along the x axis is used for the calculation
      yskip,                            ///< Every yskip voxel along the y axis is used for the calculation
      zskip;                            ///< Every zskip voxel along the z axis is used for the calculation

public:
  SimilaritySSD(void);
  virtual ~SimilaritySSD(void);

  virtual double eval(const reglib::ImagePair<int3>& images);
  virtual void setParameter(int id, double value);
  virtual void setParameter(int id, void* value);

};

}

#endif /* REGUTIL_SIMILARITYSSD_H_ */
