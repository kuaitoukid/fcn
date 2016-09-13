/// protobuf
optional SerializeParameter serialize_param;
message SerializeParameter {
  optional uint32 axis = 1;
  optional float max_value = 2 [default = 1];
}

/// hpp
#ifndef CAFFE_SERIALIZE_LAYER_HPP_
#define CAFFE_SERIALIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Ouput a serialized layer of the same input size.
 */
template <typename Dtype>
class SerializeLayer : public Layer<Dtype> {
 public:
  explicit ScaleLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Serialize"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { NOT_IMPLEMENTED; }
  
  int axis_;
  Dtype stride_value_;
  Dtype* serialize_data_;
}
}
/// cpp
#include <algorithm>
#include <vector>

#include "caffe/layer_factory.hpp"
#include "caffe/layers/serialize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void SerializeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const SerializeParameter& param = this->layer_param_.serialize_param();
  CHECK(param.has_axis()) << "axis value should be assigned";
  axis_ = param.axis();
  CHECK_GE(axis_, 0) << "axis should be >= 0";
  CHECK_LE(axis_, 3) << "axis should be < 4";
  stride_value_ = param.max_value() / bottom[0]->shape(axis_);
  // Dtype* top_data = top[0]->mutable_cpu_data();
  serialize_data_ = new Dtype[bottom[0]->count()];
  int outer_duplicate_size = 1;
  int inner_duplicate_size = 1;
  for (int i = 0; i < axis_; i ++) {
    outer_duplicate_size *= bottom[0]->shape(i);
  }
  for (int i = 3; i > axis_; i --) {
    inner_duplicate_size *= bottom[0]->shape(i);
  }
  int duplicate_data_size = inner_duplicate_size * bottom[0]->shape(axis_);
  Dtype* duplicate_data = new Dtype[duplicate_data_size];
  
  for (int n = 0; n < outer_duplicate_size; n ++) {
    for (int i = 0; i < bottom[0]->shape(axis_); i ++) {
      caffe_set(inner_duplicate_size, stride_value_ * (i + 1), duplicate_data + i * inner_duplicate_size);
    }
    caffe_copy(duplicate_data_size, duplicate_data, serialize_data_ + n * duplicate_data_size);
  }
  
  
}

template <typename Dtype>
void SerializeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SerializeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[0]->count(), serialize_data_, top[0]->mutable_cpu_data());
}
#ifdef CPU_ONLY
STUB_GPU(SerializeLayer);
#endif

INSTANTIATE_CLASS(SerializeLayer);
REGISTER_LAYER_CLASS(Serialize);
}

