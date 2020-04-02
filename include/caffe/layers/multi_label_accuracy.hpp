#ifndef CAFFE_MULTI_LABEL_ACCURACY_LAYER_HPP_
#define CAFFE_MULTI_LABEL_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
/* MultiLabelAccuracyLayer
  Note: not an actual loss layer! Does not implement backwards step.
  Computes the accuracy of a with respect to b.
*/
template <typename Dtype>
class MultiLabelAccuracyLayer : public Layer<Dtype> {
 public:
  explicit MultiLabelAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "MultiLabelAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe


#endif  // CAFFE_MULTI_LABEL_ACCURACY_LAYERS_HPP_