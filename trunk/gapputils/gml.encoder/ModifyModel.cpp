/*
 * ModifyModel.cpp
 *
 *  Created on: Apr 10, 2015
 *      Author: tombr
 */

#include "ModifyModel.h"

#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/math.hpp>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(ModifyModel)

  ReflectableBase(DefaultWorkflowElement<ModifyModel>)

  WorkflowProperty(InputModel, Input("ENN"), NotNull<Type>())
  WorkflowProperty(Filters, Input("F"))
  WorkflowProperty(Biases, Input("Bs"))
  WorkflowProperty(Weights, Input("W"))
  WorkflowProperty(Bias, Input("B"))
  WorkflowProperty(Layer)
  WorkflowProperty(Shortcut, Flag())
  WorkflowProperty(OutputModel, Output("ENN"))

EndPropertyDefinitions

ModifyModel::ModifyModel() : _Layer(0), _Shortcut(false) {
  setLabel("Modify");
}

void ModifyModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  int celayerCount = getInputModel()->cnn_encoders().size();
  int delayerCount = getInputModel()->nn_encoders().size();
  int cdlayerCount = getInputModel()->cnn_encoders().size();
  int ddlayerCount = getInputModel()->nn_encoders().size();

  if (getLayer() >= celayerCount + delayerCount + cdlayerCount + ddlayerCount) {
    dlog(Severity::Warning) << "Invalid layer specified. The given network has only " << celayerCount + delayerCount + cdlayerCount + ddlayerCount << " layers. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInputModel()));

  if (getLayer() < celayerCount) {
    dlog(Severity::Trace) << "Setting information for convolutional encoding layer " << getLayer();
    cnn_layer_t& cnn_layer = *model->cnn_encoders()[getLayer()];

    if (getFilters()) {
      if (getFilters()->size() != cnn_layer.filters().size()) {
        dlog(Severity::Warning) << "Number of filters must match between original model and given filters. Aborting!";
        return;
      }
      for (size_t i = 0; i < cnn_layer.filters().size(); ++i) {
        cnn_layer.filters()[i] = boost::make_shared<tensor_t>(*getFilters()->at(i));
      }
    } else {
      for (size_t i = 0; i < cnn_layer.filters().size(); ++i) {
        cnn_layer.filters()[i] = boost::make_shared<tensor_t>(zeros<value_t>(cnn_layer.filters()[i]->size()));
      }
    }

    if (getBiases()) {
      if (getBiases()->size() != cnn_layer.bias().size()) {
        dlog(Severity::Warning) << "Number of biases must match between original model and given biases. Aborting!";
        return;
      }
      for (size_t i = 0; i < cnn_layer.bias().size(); ++i) {
        cnn_layer.bias()[i] = boost::make_shared<tensor_t>(sum(*getBiases()->at(i)) / getBiases()->at(i)->count() * ones<value_t>(cnn_layer.bias()[i]->size()));
      }
    } else {
      for (size_t i = 0; i < cnn_layer.filters().size(); ++i) {
        cnn_layer.bias()[i] = boost::make_shared<tensor_t>(zeros<value_t>(cnn_layer.bias()[i]->size()));
      }
    }
  } else if (getLayer() < celayerCount + delayerCount) {
    dlog(Severity::Warning) << "Dense layer modifications not implemented. Aborting!";
    return;
  } else if (getLayer() < celayerCount + delayerCount + ddlayerCount) {
    dlog(Severity::Warning) << "Dense layer modifications not implemented. Aborting!";
    return;
  } else if (getLayer() < celayerCount + delayerCount + ddlayerCount + cdlayerCount) {
    const int iLayer = getLayer() - celayerCount - delayerCount - ddlayerCount;

    if (!_Shortcut) {
      dlog(Severity::Trace) << "Setting information for convolutional decoding layer " << iLayer;
      dnn_layer_t& dnn_layer = *model->dnn_decoders()[iLayer];

      if (getFilters()) {
        if (getFilters()->size() != dnn_layer.filters().size()) {
          dlog(Severity::Warning) << "Number of filters must match between original model (" << dnn_layer.filters().size() << ") and given filters (" << getFilters()->size() << "). Aborting!";
          return;
        }
        for (size_t i = 0; i < dnn_layer.filters().size(); ++i) {
          dnn_layer.filters()[i] = boost::make_shared<tensor_t>(*getFilters()->at(i));
        }
      } else {
        for (size_t i = 0; i < dnn_layer.filters().size(); ++i) {
          dnn_layer.filters()[i] = boost::make_shared<tensor_t>(zeros<value_t>(dnn_layer.filters()[i]->size()));
        }
      }

      if (getBiases()) {
        if (getBiases()->size() != 1) {
          dlog(Severity::Warning) << "Number of biases must be one. Aborting!";
          return;
        }

        dnn_layer.set_bias(sum(*getBiases()->at(0)) / getBiases()->at(0)->count() * ones<value_t>(dnn_layer.bias().size()));
      } else {
        dnn_layer.set_bias(zeros<value_t>(dnn_layer.bias().size()));
      }
    } else if (iLayer > 0) {
      dlog(Severity::Trace) << "Setting information for convolutional decoding shortcuts " << iLayer;

      dnn_layer_t& dnn_layer = *model->dnn_shortcuts()[iLayer - 1];

      if (getFilters()) {
        if (getFilters()->size() != dnn_layer.filters().size()) {
          dlog(Severity::Warning) << "Number of filters must match between original model (" << dnn_layer.filters().size() << ") and given filters (" << getFilters()->size() << "). Aborting!";
          return;
        }
        for (size_t i = 0; i < dnn_layer.filters().size(); ++i) {
          dnn_layer.filters()[i] = boost::make_shared<tensor_t>(*getFilters()->at(i));
        }
      } else {
        for (size_t i = 0; i < dnn_layer.filters().size(); ++i) {
          dnn_layer.filters()[i] = boost::make_shared<tensor_t>(zeros<value_t>(dnn_layer.filters()[i]->size()));
        }
      }

      if (getBiases()) {
        if (getBiases()->size() != 1) {
          dlog(Severity::Warning) << "Number of biases must be one. Aborting!";
          return;
        }

        dnn_layer.set_bias(sum(*getBiases()->at(0)) / getBiases()->at(0)->count() * ones<value_t>(dnn_layer.bias().size()));
      } else {
        dnn_layer.set_bias(zeros<value_t>(dnn_layer.bias().size()));
      }
    } else {
      dlog(Severity::Message) << "Given decoding layer does not have a short cut connection.";
    }
  }

  newState->setOutputModel(model);
}

} /* namespace encoder */

} /* namespace gml */
