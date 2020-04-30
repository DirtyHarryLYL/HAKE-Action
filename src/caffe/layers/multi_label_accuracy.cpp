// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_label_accuracy.hpp"
#include "caffe/util/io.hpp"

using std::max;


namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
    << "The data and label should have the same number of labels";

}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Top will contain:
  // top[0] = Sensitivity (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = Harmonic Mean of Sens and Spec, 2/(P/TP+N/TN),
  top[0]->Reshape(1, 3, 1, 1);
  
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype true_positive = 0;
  Dtype true_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label == 1) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] > 0.8);
      count_pos++;
    }
    if (label == 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0.2);
      count_neg++;
    }
  }
  DLOG(INFO) << "Sensitivity: " << (true_positive / count_pos);
  DLOG(INFO) << "Specificity: " << (true_negative / count_neg);
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " <<
    2 / ( count_pos / true_positive + count_neg / true_negative);
  top[0]->mutable_cpu_data()[0] = true_positive / count_pos;
  top[0]->mutable_cpu_data()[1] = true_negative / count_neg;
  top[0]->mutable_cpu_data()[2] =
    2 / (count_pos / true_positive + count_neg / true_negative);
  
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);
}  // namespace caffe