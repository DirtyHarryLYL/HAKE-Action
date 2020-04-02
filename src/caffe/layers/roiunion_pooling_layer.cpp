// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roiunion_pooling.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIUnionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIUnionPoolingParameter roi_pool_param = this->layer_param_.roiunion_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIUnionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIUnionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois1 = bottom[1]->cpu_data();
  const Dtype* bottom_rois2 = bottom[2]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi1_batch_ind = bottom_rois1[0];
    int roi1_start_w = round(bottom_rois1[1] * spatial_scale_);
    int roi1_start_h = round(bottom_rois1[2] * spatial_scale_);
    int roi1_end_w = round(bottom_rois1[3] * spatial_scale_);
    int roi1_end_h = round(bottom_rois1[4] * spatial_scale_);
    int roi2_batch_ind = bottom_rois2[0];
    int roi2_start_w = round(bottom_rois2[1] * spatial_scale_);
    int roi2_start_h = round(bottom_rois2[2] * spatial_scale_);
    int roi2_end_w = round(bottom_rois2[3] * spatial_scale_);
    int roi2_end_h = round(bottom_rois2[4] * spatial_scale_);

    CHECK_GE(roi1_batch_ind, 0);
    CHECK_LT(roi1_batch_ind, batch_size);

    //Union box
    int roi_start_w = min(roi1_start_w,roi2_start_w);
    int roi_start_h = min(roi1_start_h,roi2_start_h);
    int roi_end_w = max(roi1_start_w,roi2_start_w);
    int roi_end_h = max(roi1_start_h,roi2_start_h);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi1_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          Dtype maxval=0;
          int maxidx=-1;
          const int pool_index = ph * pooled_width_ + pw;
          if (!is_empty) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (batch_data[index] > top_data[pool_index]) {
                  maxval = batch_data[index];
                  maxidx = index;
                }
              }
            }
          }
          top_data[pool_index] = maxval;
          argmax_data[pool_index] = maxidx;
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois1 += bottom[1]->offset(1);
    bottom_rois2 += bottom[2]->offset(1);
  }
}

template <typename Dtype>
void ROIUnionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIUnionPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIUnionPoolingLayer);
REGISTER_LAYER_CLASS(ROIUnionPooling);

}  // namespace caffe