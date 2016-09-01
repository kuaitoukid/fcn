#include <functional>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/layers/multiscale_fuse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    
using std::min;
using std::max;

template <typename Dtype>
void MultiscaleFuseLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const MultiscaleFuseParameter& multiscale_fuse_param = this->layer_param_.multiscale_fuse_param();
  
  channels_per_scale_ = multiscale_fuse_param.channels_per_scale();
  threshold_ = multiscale_fuse_param.threshold();

  nms_need_nms_ = multiscale_fuse_param.nms_param().need_nms();
  nms_overlap_ratio_ = multiscale_fuse_param.nms_param().overlap_ratio();
  nms_top_n_ = multiscale_fuse_param.nms_param().top_n();
  nms_add_score_ = multiscale_fuse_param.nms_param().add_score(); 
  min_box_size_ = multiscale_fuse_param.min_box_size(); 

  has_detection_ = multiscale_fuse_param.has_detection();
  has_recognition_ = multiscale_fuse_param.has_recognition();
  has_regression_ = multiscale_fuse_param.has_regression();
  has_rotation_ = multiscale_fuse_param.has_rotation();
  class_num_ = multiscale_fuse_param.class_num();
  detection_channel_ = multiscale_fuse_param.detection_channel();
  regression_channel_ = multiscale_fuse_param.regression_channel();
  
  CHECK(has_detection_ && has_regression_) << "Need both detection and regression.";
  
  detection_offset_ = 0;
  recognition_offset_ = has_detection_ * detection_channel_ + detection_offset_;
  regression_offset_ = has_recognition_ * class_num_ + has_detection_ * detection_channel_ + detection_offset_;
  rotation_offset_ = has_regression_ * regression_channel_ + has_recognition_ * class_num_ + has_detection_ * detection_channel_ + detection_offset_;

  detection_data_ = NULL;
  recognition_data_ = NULL;
  regression_data_ = NULL;
  rotation_data_ = NULL;

  tested_all_ = false;
}

template <typename Dtype>
void MultiscaleFuseLayer<Dtype>::NMS(vector<Box> dense_boxes, vector<Box>& nms_boxes, 
  const Dtype overlap_ratio, const int top_n, const bool add_score) {
  if (dense_boxes.size() == 0) return;
  vector<bool> skip(dense_boxes.size(), false);
  vector<Dtype> areas(dense_boxes.size(), 0);

  std::stable_sort(dense_boxes.begin(), dense_boxes.end()); 

  for (int i = 0; i < dense_boxes.size(); i ++) {
    areas[i] = (dense_boxes[i].right - dense_boxes[i].left + 1) * (dense_boxes[i].bottom - dense_boxes[i].top + 1);
    //std::cout << "score: " << dense_boxes[i].score << ", width: " << (dense_boxes[i].right - dense_boxes[i].left + 1) << ", height: " <<
    //  (dense_boxes[i].bottom - dense_boxes[i].top + 1) << std::endl;
    //std::cout << "left: " << dense_boxes[i].left << ", top: " << dense_boxes[i].top 
    //          << ", right: " << dense_boxes[i].right << ", bottom: " << dense_boxes[i].bottom << std::endl;
  }

  for (int count = 0, i = 0; count < top_n && i < dense_boxes.size(); i ++) {
    if (skip[i]) continue;
    nms_boxes.push_back(dense_boxes[i]);
    //std::cout << "iterm: " << i << std::endl;
    count ++;
    // Suppress the significantly covered boxes
    for (int j = i + 1; j < dense_boxes.size(); j ++) {
      if (skip[j]) continue;
      // Get intersections
      Dtype inter_left = std::max(dense_boxes[i].left, dense_boxes[j].left);
      Dtype inter_top = std::max(dense_boxes[i].top, dense_boxes[j].top);
      Dtype inter_right = std::min(dense_boxes[i].right, dense_boxes[j].right);
      Dtype inter_bottom = std::min(dense_boxes[i].bottom, dense_boxes[j].bottom);
      Dtype inter_w = inter_right - inter_left + 1;
      Dtype inter_h = inter_bottom - inter_top + 1;
      if (inter_w > 0 && inter_h > 0 && areas[i] > 0 && areas[j] > 0) {
        Dtype iou = inter_w * inter_h / std::min(areas[i], areas[j]);
        if (iou > overlap_ratio) {
          skip[j] = true;
          if (add_score) {
            dense_boxes[i].score += dense_boxes[j].score;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void MultiscaleFuseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  static int kk = 0;
  std::cout << "iter: " << kk++ << ", block number: " << bottom[0]->num() << std::endl;

  //FILE* fp = fopen("heatmap.txt", "w");
  //for (int i = 0; i < bottom[0]->count(); i ++) {
  //  fprintf(fp, "%.2f ", bottom[0]->cpu_data()[i]);
  //}
  //fprintf(fp, "\n");
  //fclose(fp);
  //CHECK(false);
  

  for (int blockid = 0; blockid < bottom[0]->num(); blockid ++) {
    // image_id, scale, left, top, width, height, downsampling_ratio
    const Dtype* block_info_data = bottom[1]->cpu_data() + blockid * bottom[1]->width();
    int imid = int(block_info_data[0]);
    Dtype scale = block_info_data[1];
    int left_offset = int(block_info_data[2]);
    int top_offset = int(block_info_data[3]);
    int input_width = int(block_info_data[4]);
    int input_height = int(block_info_data[5]);
    // std::cout << scale << std::endl;
    // std::cout << "left offset: " << left_offset << ", top offset: " << top_offset << std::endl;
    //std::cout << "imid: " << imid << ", scale: " << scale << ", left: " << left_offset << ", top: " 
    //          << top_offset << ", inputw: " << input_width << ", inputh: " << input_height << std::endl;
    // bottom[0]
    int out_width = bottom[0]->width();
    int out_height = bottom[0]->height();
    int scale_num = bottom[0]->channels() / channels_per_scale_;
    Dtype down_sampling_ratio = Dtype( 1.0 * out_width / input_width + 1.0 * out_height / input_height ) / 2;
    int up_sampling_ratio = int(1 / down_sampling_ratio) / 2;

    const Dtype* bottom_data_per_sample = bottom[0]->cpu_data() + blockid * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    vector<Box> boxes;

    for (int sclid = 0; sclid < scale_num; sclid ++) {
//      std::cout << "000000000000000000000000" << std::endl;
      const Dtype* bottom_data_per_scale = bottom_data_per_sample + sclid * channels_per_scale_;
      if (has_detection_) {
        detection_data_ = bottom_data_per_scale + out_width * out_height * 0;
      }
      if (has_recognition_) {
        recognition_data_ = bottom_data_per_scale + out_width * out_height * recognition_offset_;
      }
      if (has_regression_) {
        regression_data_ = bottom_data_per_scale + out_width * out_height * regression_offset_;
      }
      if (has_rotation_) {
        rotation_data_ = bottom_data_per_scale + out_width * out_height * rotation_offset_;
      }

      for (int h = up_sampling_ratio; h < out_height - up_sampling_ratio; h ++) {
        for (int w = up_sampling_ratio; w < out_width - up_sampling_ratio; w ++) {
          int out_index = w + h * out_width;
          Dtype detection_score = detection_data_[out_index];
          if (detection_score > threshold_ || imid == -1) {
            int left_coord_index = 0;
            int top_coord_index = 0;
            int right_coord_index = 0;
            int bottom_coord_index = 0;

            Dtype left_in_image = 0;
            Dtype top_in_image = 0;
            Dtype right_in_image = 0;
            Dtype bottom_in_image = 0;

            if (regression_channel_ == 4) {
              left_coord_index = w + (h + 0 * out_height) * out_width;
              top_coord_index = left_coord_index + out_height * out_width * 1;
              right_coord_index = left_coord_index + out_height * out_width * 2;
              bottom_coord_index = left_coord_index + out_height * out_width * 3; 
              left_in_image = (( (w + regression_data_[left_coord_index]) / down_sampling_ratio + left_offset) / scale);
              top_in_image = (( (h + regression_data_[top_coord_index]) / down_sampling_ratio + top_offset) / scale);
              right_in_image = (( (w + regression_data_[right_coord_index]) / down_sampling_ratio + left_offset) / scale);
              bottom_in_image = (( (h + regression_data_[bottom_coord_index]) / down_sampling_ratio + top_offset) / scale);
            } else if (regression_channel_ == 8) {
              top_left_coord_index = w + (h + 0 * out_height) * out_width;
              left_top_coord_index = w + (h + 1 * out_height) * out_width;
              top_right_coord_index = w + (h + 2 * out_height) * out_width;
              right_top_coord_index = w + (h + 3 * out_height) * out_width;
              bottom_right_coord_index = w + (h + 4 * out_height) * out_width;
              right_bottom_coord_index = w + (h + 5 * out_height) * out_width;
              bottom_left_coord_index = w + (h + 6 * out_height) * out_width;
              left_bottom_coord_index = w + (h + 7 * out_height) * out_width;
              
              left_in_image = (((w + min(regression_data_[top_left_coord_index], 
                regression_data_[bottom_left_coord_index])) / down_sampling_ratio + left_offset) / scale);
              top_in_image = (((h + min(regression_data_[left_top_coord_index], 
                regression_data_[right_top_coord_index])) / down_sampling_ratio + top_offset) / scale);
              right_in_image = (((w + max(regression_data_[top_right_coord_index],
                regression_data_[bottom_right_coord_index])) / down_sampling_ratio + left_offset) / scale);
              bottom_in_image = (((h + max(regression_data_[left_bottom_coord_index],
                regression_data_[right_bottom_coord_index])) / down_sampling_ratio + top_offset) / scale);
            }

            vector<Dtype> cat_prob;
            if (has_recognition_) {
              for (int catid = 0; catid < class_num_; catid ++) {
                int cat_index = w + (h + catid * out_height) * out_width;
                cat_prob.push_back(recognition_data_[cat_index]);
              }
            }
            //std::cout << "1:, " << regression_data_[left_coord_index] << ",2: " << regression_data_[top_coord_index] << ",3: " <<
            //  regression_data_[right_coord_index] << ",4: " << regression_data_[bottom_coord_index] << std::endl;
            // std::cout << "downsampling_ratio: " << down_sampling_ratio << ", scale: " << scale << std::endl;
            //std::cout << "left: " << left_in_image << ",top: " << top_in_image << ",right: " << right_in_image << ",bottom: " << bottom_in_image << std::endl;
            //std::cout << "Width: " << right_in_image - left_in_image << ", " << "Height: " << bottom_in_image - top_in_image << std::endl;
            //if ((right_in_image - left_in_image < min_box_size_ || bottom_in_image - top_in_image < min_box_size_) && imid != -1) {
            //  continue;
            //}
//      std::cout << "222222222222222222222" << std::endl;
            Box tmpbox = Box(detection_score, cat_prob, left_in_image, top_in_image, right_in_image, bottom_in_image, 
              (w + left_offset * down_sampling_ratio) / scale, (h + top_offset * down_sampling_ratio) / scale, scale);

//      std::cout << "33333333333333333333" << std::endl;
            // std::cout << "fx: " << (w + left_offset * down_sampling_ratio) / scale << ", fy: " << (h + top_offset * down_sampling_ratio) / scale << std::endl;

            if (imid == -1) {
              vector<Box> tmp_nms_boxes;
			  if (dense_boxes_.size() > 0) {
                NMS(dense_boxes_[dense_boxes_.size() - 1], tmp_nms_boxes, nms_overlap_ratio_, nms_top_n_, nms_add_score_);
              }
              nms_boxes_.push_back(tmp_nms_boxes);
              // END OF TEST 
              FILE* fp = fopen("result.txt", "w");
              for (int re_im_id = 0; re_im_id < nms_boxes_.size(); re_im_id ++) {
                fprintf(fp, "image %d\n", re_im_id);
                for (int re_bid = 0; re_bid < nms_boxes_[re_im_id].size(); re_bid ++) {
                  int maxid = 0;
                  Dtype maxprob = 0;
                  for (int catid = 0; catid < class_num_ && has_recognition_; catid ++) {
                    if (maxprob < nms_boxes_[re_im_id][re_bid].cat_prob[catid]) {
                      maxprob = nms_boxes_[re_im_id][re_bid].cat_prob[catid];
                      maxid = catid;
                    }
                  }
                  fprintf(fp, "%.2f %.2f %.2f %.2f %d %.2f\n", 
                    nms_boxes_[re_im_id][re_bid].left, nms_boxes_[re_im_id][re_bid].top, 
                    nms_boxes_[re_im_id][re_bid].right, nms_boxes_[re_im_id][re_bid].bottom,
                    maxid, maxprob);
                }
              }
              fclose(fp);
              // exit(0);
              fp = fopen("origion_result.txt", "w");
              for (int re_im_id = 0; re_im_id < dense_boxes_.size(); re_im_id ++) {
                fprintf(fp, "image %d\n", re_im_id);
                for (int re_bid = 0; re_bid < dense_boxes_[re_im_id].size(); re_bid ++) {

                  int maxid = 0;
                  Dtype maxprob = 0;
                  for (int catid = 0; catid < class_num_ && has_recognition_; catid ++) {
                    if (maxprob < dense_boxes_[re_im_id][re_bid].cat_prob[catid]) {
                      maxprob = dense_boxes_[re_im_id][re_bid].cat_prob[catid];
                      maxid = catid;
                    }
                  }


                  fprintf(fp, "%.2f %.2f %.2f %.2f %.2f %d %d %f %d %f\n", 
                    dense_boxes_[re_im_id][re_bid].score,
                    dense_boxes_[re_im_id][re_bid].left, dense_boxes_[re_im_id][re_bid].top, 
                    dense_boxes_[re_im_id][re_bid].right, dense_boxes_[re_im_id][re_bid].bottom,
                    int(dense_boxes_[re_im_id][re_bid].fx), int(dense_boxes_[re_im_id][re_bid].fy),
                    dense_boxes_[re_im_id][re_bid].scale, maxid, maxprob);
                }
              }
              fclose(fp);
              tested_all_ = true;
              exit(0);
            }
            else if (imid >= dense_boxes_.size()) {
//      std::cout << "9999999999999999999999999999999999999" << std::endl;
              int tmp_densebox_size = dense_boxes_.size();
              for (int addid = 0; addid < imid - tmp_densebox_size; addid ++) {
                dense_boxes_.push_back(boxes);
              }
              boxes.push_back(tmpbox);
              dense_boxes_.push_back(boxes);
              if (imid > 0 && nms_need_nms_) {
                // perform nms for dense_boxes_[imid - 1]
                vector<Box> tmp_nms_boxes;
                NMS(dense_boxes_[imid - 1], tmp_nms_boxes, nms_overlap_ratio_, nms_top_n_, nms_add_score_);
                nms_boxes_.push_back(tmp_nms_boxes);
              }
            }
            else {
//      std::cout << "888888888888888888888888888888888" << std::endl;
              dense_boxes_[imid].push_back(tmpbox);
            }
          }
        }
      }
    }
    // std::cout << "blockid: " << blockid << std::endl;
  }
}

INSTANTIATE_CLASS(MultiscaleFuseLayer);
REGISTER_LAYER_CLASS(MultiscaleFuse);

}  // namespace caffe
