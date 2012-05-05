/*
 * Pooling.cpp
 *
 *  Created on: Apr 19, 2012
 *      Author: tombr
 */

#include "Pooling.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <cmath>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

DefineEnum(PoolingDirection)

int Pooling::inputId;

BeginPropertyDefinitions(Pooling)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputTensors, Input("T"), Volatile(), ReadOnly(), Observe(inputId = PROPERTY_ID))
  DefineProperty(Model, Input("M"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(OutputTensors, Output("T"), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  ReflectableProperty(Direction, Observe(PROPERTY_ID))
  DefineProperty(Auto, Observe(PROPERTY_ID))

EndPropertyDefinitions

Pooling::Pooling() : _Direction(PoolingDirection::Encode), _Auto(false), data(0) {
  WfeUpdateTimestamp
  setLabel("Pooling");

  Changed.connect(capputils::EventHandler<Pooling>(this, &Pooling::changedHandler));
}

Pooling::~Pooling() {
  if (data)
    delete data;
}

void Pooling::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
    writeResults();
  }
}

void Pooling::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Pooling();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getModel() || !getInputTensors()) {
    return;
  }

  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getInputTensors();
  ConvRbmModel& model = *getModel();
  int poolingSize = model.getPoolingBlockSize();
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(new std::vector<boost::shared_ptr<tensor_t> >());

  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (getDirection() == PoolingDirection::Encode) {
      tensor_t& input = *inputs[i];
      const tensor_t::dim_t& size = input.size();
      assert(size[0] % poolingSize == 0);
      assert(size[1] % poolingSize == 0);

      boost::shared_ptr<tensor_t> output(new tensor_t(size[0] / poolingSize,
          size[1] / poolingSize, size[2] * 3));
      const tensor_t::dim_t& size2 = output->size();

      for (unsigned z = 0; z < size[2]; ++z) {
        for (unsigned y = 0; y < size[1]; y += poolingSize) {
          for (unsigned x = 0; x < size[0]; x += poolingSize) {
            unsigned xmax = 0, ymax = 0;
            tensor_t::value_t maxValue = input.data()[(z * size[1] + y) * size[0] + x];
            for (int dy = 0; dy < poolingSize; ++dy) {
              for (int dx = 0; dx < poolingSize; ++dx) {
                tensor_t::value_t value = input.data()[(z * size[1] + y + dy) * size[0] + x + dx];
                if (value > maxValue) {
                  maxValue = value;
                  xmax = dx;
                  ymax = dy;
                }
              }
            }
            output->data()[(z * 3 * size2[1] + y / poolingSize) * size2[0] + x / poolingSize] = maxValue;
            output->data()[((z * 3 + 1) * size2[1] + y / poolingSize) * size2[0] + x / poolingSize] =
                (tensor_t::value_t)xmax / (tensor_t::value_t)(poolingSize - 1);
            output->data()[((z * 3 + 2) * size2[1] + y / poolingSize) * size2[0] + x / poolingSize] =
                (tensor_t::value_t)ymax / (tensor_t::value_t)(poolingSize - 1);
          }
        }
      }

      outputs->push_back(output);
    } else {
      tensor_t& input = *inputs[i];
      const tensor_t::dim_t& size = input.size();
      assert(size[2] % 3 == 0);

      boost::shared_ptr<tensor_t> output(new tensor_t(size[0] * poolingSize,
          size[1] * poolingSize, size[2] / 3));
      const tensor_t::dim_t& size2 = output->size();

      for (unsigned z = 0; z < size[2]; z += 3) {
        for (unsigned y = 0; y < size[1]; ++y) {
          for (unsigned x = 0; x < size[0]; ++x) {
            tensor_t::value_t maxValue = input.data()[(z * size[1] + y) * size[0] + x];
            unsigned xmax = input.data()[((z + 1) * size[1] + y) * size[0] + x] * (poolingSize - 1);
            unsigned ymax = input.data()[((z + 2) * size[1] + y) * size[0] + x] * (poolingSize - 1);
            xmax = max(0, min(poolingSize - 1, xmax));
            ymax = max(0, min(poolingSize - 1, ymax));
            output->data()[(z / 3 * size2[1] + y * poolingSize + ymax) * size2[0] + x * poolingSize + xmax] = maxValue;
          }
        }
      }

      outputs->push_back(output);

    }
    if (monitor) monitor->reportProgress(i * 100 / inputs.size());
  }

  data->setOutputTensors(outputs);
}

void Pooling::writeResults() {
  if (!data)
    return;

  setOutputTensors(data->getOutputTensors());
}

}

}
