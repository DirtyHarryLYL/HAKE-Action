#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mil_layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void MILLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //CHECK_EQ(bottom.size(), 1) << "MIL Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "MIL Layer takes a single blob as output.";
  channels_ = bottom[0]->channels();
  context_per_roi_ = this->layer_param_.mil_param().context_per_roi();
  top[0]->Reshape(1, channels_, 1, 1);
}
template <typename Dtype>
void MILLayer<Dtype>::Reshape(
     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  context_per_roi_ = this->layer_param_.mil_param().context_per_roi();
  //if (context_per_roi_ == 0) {
    // context_per_roi_ = bottom[1]->cpu_data()[0];
     //std::cout<<context_per_roi_<<std::endl;
  //}
  //num_rois_ = bottom[0]->num()/context_per_roi_;
  channels_ = bottom[0]->channels();
  top[0]->Reshape(1, channels_, 1, 1);
}

template <typename Dtype>
void MILLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (context_per_roi_ == 0)
      context_per_roi_ = bottom[1]->cpu_data()[0];
  num_rois_ = bottom[0]->num()/context_per_roi_;
  top[0]->Reshape(num_rois_, channels_, 1, 1);

  Dtype* top_data = top[0]->mutable_cpu_data();
  // Code to compute the image probabilities from box probabilities
  // std::cout<<context_per_roi_<<" "<<num_rois_<<" "<<channels_<<std::endl;
  // For now just substitute the max probability instead of noisy OR
  for(int k = 0; k < channels_; k++){
    for(int i = 0; i < num_rois_; i++){
      Dtype prob, max_prob; 
      
      prob = -FLT_MAX; 
      for(int j = 0; j < context_per_roi_; j++){
        prob = max(prob, bottom_data[(i*context_per_roi_ + j)*channels_+k]);
      }
      top_data[i*channels_ + k] = prob;
    }
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if(propagate_down[0]){
    // All the gradient goes to the bow with the probability equal to the 
    // probability in the top pf the layer!
    for(int k = 0; k < channels_; k++){
      for(int i = 0; i < num_rois_; i++){
        for(int j = 0; j < context_per_roi_; j++){
          
          bottom_diff[(i*context_per_roi_ + j)*channels_+k] =
            top_diff[i*channels_ + k] * 
              (top_data[i*channels_ + k] == bottom_data[(i*context_per_roi_ + j)*channels_+k]);
            
        }
      }
    }
  }
}

INSTANTIATE_CLASS(MILLayer);
REGISTER_LAYER_CLASS(MIL);

}  // namespace caffe
