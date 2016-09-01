#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/densebox_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
DenseboxDataLayer<Dtype>::~DenseboxDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //scale_base_ = this->layer_param_.densebox_data_param().scale_base();
  for (int i = 0; i < this->layer_param_.densebox_data_param().scale_base_size(); i ++)
    scale_base_.push_back( Dtype( pow(2.0, this->layer_param_.densebox_data_param().scale_base(i)) ) );

  CHECK_GT(scale_base_.size(), 0) << "scale base size < 1";
  scale_positive_upper_bounder_ = this->layer_param_.densebox_data_param().scale_positive_upper_bounder();
  scale_positive_upper_bounder_ = Dtype(pow(2.0, scale_positive_upper_bounder_));
  
  scale_positive_lower_bounder_ = this->layer_param_.densebox_data_param().scale_positive_lower_bounder();
  scale_positive_lower_bounder_ = Dtype(pow(2.0, scale_positive_lower_bounder_));
  
  scale_ignore_upper_bounder_ = this->layer_param_.densebox_data_param().scale_ignore_upper_bounder();
  scale_ignore_upper_bounder_ = Dtype(pow(2.0, scale_ignore_upper_bounder_));
  
  scale_ignore_lower_bounder_ = this->layer_param_.densebox_data_param().scale_ignore_lower_bounder();
  scale_ignore_lower_bounder_ = Dtype(pow(2.0, scale_ignore_lower_bounder_));
  
  scale_upper_limit_ = this->layer_param_.densebox_data_param().scale_upper_limit();
  // scale_upper_limit_ = Dtype(pow(2.0, scale_upper_limit_));
  
  scale_lower_limit_ = this->layer_param_.densebox_data_param().scale_lower_limit();
  // scale_lower_limit_ = Dtype(pow(2.0, scale_lower_limit_));
  
  input_height_ = this->layer_param_.densebox_data_param().input_height();
  input_width_ = this->layer_param_.densebox_data_param().input_width();
  // heat_map_a_ = this->layer_param_.densebox_data_param().heat_map_a();
  // heat_map_b_ = this->layer_param_.densebox_data_param().heat_map_b();
  out_height_ = this->layer_param_.densebox_data_param().out_height();
  out_width_ = this->layer_param_.densebox_data_param().out_width();
  num_anno_points_per_instance_ = this->layer_param_.densebox_data_param().num_anno_points_per_instance();
  ignore_bound_size_ = this->layer_param_.densebox_data_param().ignore_bound_size();
  CHECK_GT(input_height_, 0) << "input height < 1";
  CHECK_GT(input_width_, 0) << "input width < 1";
  // CHECK_GT(heat_map_a_, 0) << "heatmap_a < 1";
  CHECK_GT(out_height_, 0) << "output height < 1";
  CHECK_GT(out_width_, 0) << "output width < 1";
  CHECK_GT(num_anno_points_per_instance_, 0) << "annotation points number < 1";
  single_thread_ = this->layer_param_.densebox_data_param().single_thread();

  pos_samples_source_ = this->layer_param_.densebox_data_param().pos_samples_source();
  neg_samples_source_ = this->layer_param_.densebox_data_param().neg_samples_source();
  pos_img_folder_ = this->layer_param_.densebox_data_param().pos_img_folder();
  neg_img_folder_ = this->layer_param_.densebox_data_param().neg_img_folder();
  neg_ratio_ = this->layer_param_.densebox_data_param().neg_ratio();
  if (neg_ratio_ > 1) neg_ratio_ = 1;
  else if (neg_ratio_ < 0) neg_ratio_ = 0;

  batch_size_ = this->layer_param_.densebox_data_param().batch_size();
  pos_batch_size_ = max(int(batch_size_ * (1 - neg_ratio_)), 0);
  if (neg_ratio_ == 0) pos_batch_size_ = batch_size_;
  // std::cout << "pos_size: " << pos_batch_size_ << std::endl;
  // CHECK_EQ(1, 3);
  neg_batch_size_ = batch_size_ - pos_batch_size_;
  shuffle_ = this->layer_param_.densebox_data_param().shuffle();
  // mean_value_ = this->layer_param_.densebox_data_param().mean_value(); // vector
  for (int i = 0; i < this->layer_param_.densebox_data_param().mean_value_size(); i ++) 
    mean_value_.push_back(this->layer_param_.densebox_data_param().mean_value(i));
  CHECK(mean_value_.size() == 1 || mean_value_.size() == 3) << "mean value number invalid: either 1 or 3";
  pixel_scale_ = this->layer_param_.densebox_data_param().pixel_scale();

  roi_center_point_ = this->layer_param_.densebox_data_param().roi_center_point();
  // key_point_ = this->layer_param_.densebox_data_param().key_point();
  for (int i = 0; i < this->layer_param_.densebox_data_param().key_point_size(); i ++)
    key_point_.push_back(this->layer_param_.densebox_data_param().key_point(i));
  min_output_pos_radius_ = this->layer_param_.densebox_data_param().min_output_pos_radius();
  bbox_height_ = this->layer_param_.densebox_data_param().bbox_height();
  bbox_width_ = this->layer_param_.densebox_data_param().bbox_width();
  bbox_size_norm_type_ = this->layer_param_.densebox_data_param().bbox_size_norm_type();
  ignore_margin_ = this->layer_param_.densebox_data_param().ignore_margin();
  bbox_valid_dist_ratio_ = this->layer_param_.densebox_data_param().bbox_valid_dist_ratio();
  sample_type_point_ = this->layer_param_.densebox_data_param().sample_type_point();
  CHECK_GT(min_output_pos_radius_, 0) << "minimum output positive radius < 1";
  CHECK_GT(bbox_height_, 0) << "bounding box height < 1";
  CHECK_GT(bbox_width_, 0) << "bounding box width < 1";
  CHECK_GT(bbox_valid_dist_ratio_, 0) << "bbox_valid_dist_ratio < 0";
  
  need_detection_ = this->layer_param_.densebox_data_param().need_detection();
  need_recognition_ = this->layer_param_.densebox_data_param().need_recognition();
  need_rotation_ = this->layer_param_.densebox_data_param().need_rotation();
  need_regression_ = this->layer_param_.densebox_data_param().need_regression();
  need_ignore_ = this->layer_param_.densebox_data_param().need_ignore();
  class_index_point_ = this->layer_param_.densebox_data_param().class_index_point();
  rotation_point_ = this->layer_param_.densebox_data_param().rotation_point();
  if (!(need_detection_ || need_recognition_)) {
    need_detection_ = true;
  }

  /// load filenames and ground truths 
  std::ifstream pos_infile(pos_samples_source_.c_str());
  std::ifstream neg_infile(neg_samples_source_.c_str());
  pos_sample_id_ = 0;
  neg_sample_id_ = 0;
  // read positive file
  std::string line;
  int pos_num = 0;
  while (std::getline(pos_infile, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string filename;
    iss >> filename;
    filename = pos_img_folder_ + filename;
    vector<Dtype> coords;
    Dtype coord;
    while (iss >> coord) {
      coords.push_back(coord);
    }
    if (coords.size() % (num_anno_points_per_instance_ * 2) != 0) {
      std::cout << "Invalid annotation image: " << filename << ", coords.size: " << coords.size() / 2 << ", annot: " << num_anno_points_per_instance_ << std::endl;
      continue;
    }
    pair<std::string, vector<Dtype> > sample = make_pair(filename, coords);
    pos_samples_.push_back(sample);
    pos_num ++;
  }
  LOG(INFO) << "Positive sample number: " << pos_num;
  // read negative file
  int neg_num = 0;
  while (std::getline(neg_infile, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string filename;
    iss >> filename;
    filename = neg_img_folder_ + filename;
    vector<Dtype> coords;
    Dtype coord;
    while (iss >> coord) {
      coords.push_back(coord);
    }
    if (coords.size() % (num_anno_points_per_instance_ * 2) != 0) {
      std::cout << "Invalid annotation image: " << filename << std::endl;
      continue;
    }
    pair<std::string, vector<Dtype> > sample = make_pair(filename, coords);  
    neg_samples_.push_back(sample);
    neg_num ++;
  }
  LOG(INFO) << "Negative sample number: " << neg_num;
  LOG(INFO) << "Total sample number: " << pos_num + neg_num;
  // Initial random seed
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  // Shuffle images
  if (shuffle_) {
    //LOG(INFO) << "Initializing random seed and shuffling data";
    //const unsigned int prefetch_rng_seed = caffe_rng_rand();
    //prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    // ShufflePosImages();
    // ShuffleNegImages();
    ShuffleSamples(pos_samples_);
    ShuffleSamples(neg_samples_);
  }
  /// Initialize top blobs
  // reshape top[0] to (batchsize, 3, input_height, input_width)
  top[0]->Reshape(batch_size_, 3, input_height_, input_width_);
  // reshape top[1] to (batchsize, ch, out_height, out_width)
  label_channel_ = 0;
  if (need_detection_) { 
    detection_channel_offset_ = label_channel_;
    label_channel_ ++;
  }
  if (need_recognition_) {
    recognition_channel_offset_ = label_channel_;
    label_channel_ ++;
  }
  //if (need_ignore_) {//(need_detection_ || need_recognition_) {
  //  ignore_channel_offset_ = label_channel_;
  //  label_channel_ ++;
  //}
  if (need_rotation_) {
    rotation_channel_offset_ = label_channel_;
    label_channel_ ++;
  }
  if (need_regression_) {
    regression_channel_offset_ = label_channel_;
    label_channel_ += key_point_.size() * 2;
  }
  label_channel_ *= scale_base_.size();
  top[1]->Reshape(batch_size_, label_channel_, out_height_, out_width_);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size_, 3, input_height_, input_width_);
    this->prefetch_[i].label_.Reshape(batch_size_, label_channel_, out_height_, out_width_);
  }
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::ShuffleSamples(vector<std::pair<std::string, vector<Dtype> > >& samples) {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(samples.begin(), samples.end(), prefetch_rng);
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::ShuffleGTs(std::pair<std::string, vector<Dtype> >& sample) {
  vector<Dtype>& coords = sample.second;
  int instance_num = coords.size() / num_anno_points_per_instance_ / 2;
  unsigned int swap_id = caffe_rng_rand() % instance_num;
  if (swap_id == 0) return;
  // swap instances
  for (int i = 0; i < num_anno_points_per_instance_ * 2; i ++) {
    Dtype tmp = coords[i];
    coords[i] = coords[i + swap_id * num_anno_points_per_instance_ * 2];
    coords[i + swap_id * num_anno_points_per_instance_ * 2] = tmp;
  }
}


template <typename Dtype>
unsigned int DenseboxDataLayer<Dtype>::PrefetchRand()
{
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
float DenseboxDataLayer<Dtype>::PrefetchRandFloat()
{
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return  PrefetchRand() / static_cast<float>(prefetch_rng->max());
}

template <typename Dtype>
Dtype DenseboxDataLayer<Dtype>::Distance(Dtype x1, Dtype y1, Dtype x2, Dtype y2) {
  return Dtype(sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::SetSampleScales(vector<std::pair<float, float> >& sample_scales,
  vector<std::pair<std::string, vector<Dtype> > > samples) {
  for (int i = 0; i < samples.size(); i ++) {
    // check data type
    vector<Dtype> coords = samples[i].second;
    if (coords[sample_type_point_ * 2] > 0) { // positive or ignore
      Dtype tmp_scale = Dtype(pow(2.0, PrefetchRandFloat() * (scale_upper_limit_ - scale_lower_limit_) + scale_lower_limit_));
      sample_scales[i].first *= tmp_scale;
      // sample_scales[i].first *= Dtype(pow(2.0, PrefetchRandFloat() * (scale_upper_limit_ - scale_lower_limit_) + scale_lower_limit_));
      sample_scales[i].second *= tmp_scale;
      // sample_scales[i].second *= Dtype(pow(2.0, PrefetchRandFloat() * (scale_upper_limit_ - scale_lower_limit_) + scale_lower_limit_));
      
      Dtype ori_bbox_height = 0;
      Dtype ori_bbox_width = 0;
      if (key_point_.size() == 1) {
        Dtype center_x = coords[roi_center_point_ * 2];
        Dtype center_y = coords[roi_center_point_ * 2 + 1];
        ori_bbox_width = 2 * abs(coords[key_point_[0] * 2] - center_x);
        ori_bbox_height = 2 * abs(coords[key_point_[0] * 2 + 1] - center_y);
      } else if (key_point_.size() == 2) {
        ori_bbox_width = abs(coords[key_point_[1] * 2] - coords[key_point_[0] * 2]);
        ori_bbox_height = abs(coords[key_point_[1] * 2 + 1] - coords[key_point_[0] * 2 + 1]);
      } else if (key_point_.size() == 4) {
        Dtype line01 = Distance(coords[key_point_[0] * 2], coords[key_point_[0] * 2 + 1], coords[key_point_[1] * 2], coords[key_point_[1] * 2 + 1]);
        Dtype line12 = Distance(coords[key_point_[1] * 2], coords[key_point_[1] * 2 + 1], coords[key_point_[2] * 2], coords[key_point_[2] * 2 + 1]);
        Dtype line23 = Distance(coords[key_point_[2] * 2], coords[key_point_[2] * 2 + 1], coords[key_point_[3] * 2], coords[key_point_[3] * 2 + 1]);
        Dtype line30 = Distance(coords[key_point_[3] * 2], coords[key_point_[3] * 2 + 1], coords[key_point_[0] * 2], coords[key_point_[0] * 2 + 1]);
        ori_bbox_width = (line01 + line23) / 2;
        ori_bbox_height = (line12 + line30) / 2;
      } else {
        LOG(ERROR) << "Invalid regression point number.";
      }
      float norm_rate = 0;
      switch(bbox_size_norm_type_){
      case DenseboxDataParameter_BBoxSizeNormType_HEIGHT:
        norm_rate = bbox_height_ / ori_bbox_height;
        break;
      case DenseboxDataParameter_BBoxSizeNormType_WIDTH:
        norm_rate = bbox_width_ / ori_bbox_width;
        break;
      case DenseboxDataParameter_BBoxSizeNormType_DIAG:
        norm_rate = sqrt(bbox_height_ * bbox_height_ + bbox_width_ * bbox_width_) / 
          sqrt(ori_bbox_width * ori_bbox_width + ori_bbox_height * ori_bbox_height);
        break;
      default:
        LOG(ERROR) << "Unknown box norm type.";
      }
      sample_scales[i].first *= norm_rate;
      sample_scales[i].second *= norm_rate;

      // std::cout << "pos: xscale: " << sample_scales[i].first << ", yscale: " << sample_scales[i].second << std::endl;
    }
    else { // negative ... need refinement
      sample_scales[i].first = (Dtype)1.0; // Dtype(pow(2.0, 1 * (PrefetchRandFloat() - 0.5)));
      sample_scales[i].second = sample_scales[i].first;
      // std::cout << "neg: xscale: " << sample_scales[i].first << ", yscale: " << sample_scales[i].second << std::endl;
      // sample_scales[i].second = Dtype(pow(2.0, 6 * (PrefetchRandFloat() - 0.5)));
      // std::cout<< sample_scales[i].first << ", " << sample_scales[i].second << std::endl;
    }
  }
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::ScaleImagesAndCoords(vector<std::pair<float, float> > sample_scales, 
  vector<std::pair<std::string, vector<Dtype> > >& samples, vector<cv::Mat>& cv_imgs) {
  for (int i = 0; i < samples.size(); i ++) {
    // scale image
    cv::Mat ori_cv_img = cv::imread(samples[i].first, CV_LOAD_IMAGE_COLOR); // CV_LOAD_IMAGE_COLOR or CV_LOAD_IMAGE_GRAYSCALE
    CHECK(ori_cv_img.data) << "Could not load " << samples[i].first;
    float x_scale = sample_scales[i].first;
    float y_scale = sample_scales[i].second;

    // std::cout << "xscale: " << x_scale << ", yscale: " << y_scale << ", rows: " << int(ori_cv_img.rows * y_scale) << ", cols: " << int(ori_cv_img.cols * x_scale) << std::endl;

    cv::Mat cv_img(max(int(ori_cv_img.rows * y_scale), 1), max(int(ori_cv_img.cols * x_scale), 1), CV_8UC3, cv::Scalar(0));
    cv::resize(ori_cv_img, cv_img, cv_img.size(), 0, 0, CV_INTER_LINEAR);
    cv_imgs.push_back(cv_img);
    // scale coords
    vector<Dtype>& coords = samples[i].second;
    if (coords[sample_type_point_ * 2] <= 0) {
      // std::cout << "xscale: " << x_scale << ", yscale: " << y_scale << ", rows: " << int(ori_cv_img.rows * y_scale) << ", cols: " << int(ori_cv_img.cols * x_scale) << std::endl;
      // coords[roi_center_point_ * 2] = PrefetchRand() % cv_img.cols;
      // coords[roi_center_point_ * 2 + 1] = PrefetchRand() % cv_img.rows;
      coords[roi_center_point_ * 2] = cv_img.cols / 2;
      coords[roi_center_point_ * 2 + 1] = cv_img.rows / 2;
      // std::cout << "centerx: " << coords[roi_center_point_ * 2] << ", centery: " << coords[roi_center_point_ * 2 + 1] 
      //          << ", imw: " << cv_img.cols << ", imh: " << cv_img.rows << std::endl;
    }
    else {
      int instance_num = coords.size() / num_anno_points_per_instance_ / 2;
      for (int j = 0; j < instance_num; j ++) {
        coords[roi_center_point_ * 2 + num_anno_points_per_instance_ * 2 * j] *= x_scale;
        coords[roi_center_point_ * 2 + 1 + num_anno_points_per_instance_ * 2 * j] *= y_scale;

        for (int k = 0; k < key_point_.size(); k ++) {
          coords[key_point_[k] * 2 + num_anno_points_per_instance_ * 2 * j] *= x_scale; 
          coords[key_point_[k] * 2 + 1 + num_anno_points_per_instance_ * 2 * j] *= y_scale; 
        }
      }
    }
  }
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::SetCropAndPad(vector<cv::Mat>& cv_imgs, vector<std::pair<std::string, vector<Dtype> > >& samples) {
  for (int i = 0; i < samples.size(); i ++) {
    cv::Mat ori_img = cv_imgs[i];
    cv::Mat dst_img(input_height_, input_width_, CV_8UC3, cv::Scalar(0));
    vector<Dtype>& coords = samples[i].second;

    const int ori_height = cv_imgs[i].rows;
    const int ori_width = cv_imgs[i].cols;
    
    // const int dst_center_x = int(input_width_ / 2);
    int dst_center_x = int(PrefetchRand() % int(input_width_ - 2 * bbox_width_) + bbox_width_);
    if (coords[sample_type_point_ * 2] <= 0) {
      dst_center_x = int(input_width_ / 2);
    }
    // const int dst_center_y = int(input_height_ / 2);
    int dst_center_y = int(PrefetchRand() % int(input_width_ - 2 * bbox_height_) + bbox_height_);
    if (coords[sample_type_point_ * 2] <= 0) {
      dst_center_y = int(input_height_ / 2);
    }

    int roi_center_x = int(coords[roi_center_point_ * 2]);
    int roi_center_y = int(coords[roi_center_point_ * 2 + 1]);

    // char fname[20];
    // sprintf(fname, "roi%d_%d_%d.jpg", i, roi_center_x, roi_center_y);
    // cv::imwrite(fname, cv_imgs[i]);
    //if (coords[sample_type_point_ * 2] <= 0) { // randomly set roi center
    //}
    const int offset_x = roi_center_x - dst_center_x;
    const int offset_y = roi_center_y - dst_center_y;
    // fill dst_img
    for (int h = 0; h < input_height_; h ++) {
      for (int w = 0; w < input_width_; w ++) {
        int ori_w = w + offset_x;
        int ori_h = h + offset_y;
        if (ori_w >= 0 && ori_h >= 0 && ori_w < ori_width && ori_h < ori_height) {
          dst_img.at<cv::Vec3b>(h, w)[0] = ori_img.at<cv::Vec3b>(ori_h, ori_w)[0];
          dst_img.at<cv::Vec3b>(h, w)[1] = ori_img.at<cv::Vec3b>(ori_h, ori_w)[1];
          dst_img.at<cv::Vec3b>(h, w)[2] = ori_img.at<cv::Vec3b>(ori_h, ori_w)[2];
        }
      }
    }
    cv_imgs[i] = dst_img; 
    // move key points
    int instance_num = coords.size() / num_anno_points_per_instance_ / 2;
    if (coords[sample_type_point_ * 2] > 0) { // positive or ignore
      for (int bid = 0; bid < instance_num; bid ++) {
        coords[bid * num_anno_points_per_instance_ * 2 + roi_center_point_ * 2] -= offset_x;
        coords[bid * num_anno_points_per_instance_ * 2 + roi_center_point_ * 2 + 1] -= offset_y;
        for (int j = 0; j < key_point_.size(); j ++) {
          coords[bid * num_anno_points_per_instance_ * 2 + key_point_[j] * 2] -= offset_x;
          coords[bid * num_anno_points_per_instance_ * 2 + key_point_[j] * 2 + 1] -= offset_y;
        }
      }
    }
  }
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::SetTopSamples(Dtype* top_data, Dtype* top_label, 
  vector<std::pair<std::string, vector<Dtype> > > samples, vector<cv::Mat> cv_imgs) {
  int img_channel = cv_imgs[0].channels();

  memset(top_data, 0, sizeof(Dtype) * samples.size() * img_channel * input_height_ * input_width_);
  memset(top_label, 0, sizeof(Dtype) * samples.size() * label_channel_ * out_height_ * out_width_);

  vector<Dtype> mean_v;
  if (img_channel == 1) 
  {
    Dtype tmp = 0;
    for (int i = 0; i < mean_value_.size(); i ++) tmp += mean_value_[i];
    tmp /= mean_value_.size();
    mean_v.push_back(tmp);
  } else if (img_channel == 3) {
    mean_v.push_back(mean_value_[0 % mean_value_.size()]);
    mean_v.push_back(mean_value_[1 % mean_value_.size()]);
    mean_v.push_back(mean_value_[2 % mean_value_.size()]);
  } else LOG(ERROR) << "Invalid image channel";

  const int label_channel_per_scale = label_channel_ / scale_base_.size();
  float downsample_ratio = (out_height_ / (input_height_ + 0.0f) + out_width_ / (input_width_ + 0.0f)) / 2;

  for (int i = 0; i < samples.size(); i ++) {
    // get top data
    // char fname[20];
    // static int kk = 0;
    // sprintf(fname, "roi%d.jpg", kk ++);
    // cv::imwrite(fname, cv_imgs[i]);
    // FILE* fp = fopen("img.txt", "w");
    // int cnt = 0;
    for (int c = 0; c < img_channel; c ++) {
      for (int h = 0; h < input_height_; h ++) {
        for (int w = 0; w < input_width_; w ++) {
          int top_data_index = (c * input_height_ + h) * input_width_ + w;
          Dtype pixel = Dtype(cv_imgs[i].at<cv::Vec3b>(h, w)[c]);
          top_data[top_data_index] = (pixel - mean_v[c]) * pixel_scale_;
          //std::cout  << pixel << std::endl;
          //fprintf(fp, "%.3f ", pixel);
          //cnt ++;
        }
      }
    }
    //fclose(fp);
    //CHECK_EQ(1, 3) << "ENd ..." << cnt;
    // set ignore label to 1-matrix but chosen_scale_channel for positive samples
    vector<Dtype> coords = samples[i].second;
    if (coords[sample_type_point_ * 2] > 0) { // positive or ignore
      int instance_num = coords.size() / num_anno_points_per_instance_ / 2;
      for (int bid = 0; bid < instance_num; bid ++) { 
        const int center_w = int(coords[bid * num_anno_points_per_instance_ * 2 + roi_center_point_ * 2]);
        const int center_h = int(coords[bid * num_anno_points_per_instance_ * 2 + roi_center_point_ * 2 + 1]);
        if (center_w < 0 || center_h < 0 || center_w >= input_width_ || center_h >= input_height_) continue;
        const int center_w_label = int(center_w * downsample_ratio);
        const int center_h_label = int(center_h * downsample_ratio);
        int bb_width = 0;
        int bb_height = 0;

        // int tl_to_center_x = 0;
        // int tl_to_center_y = 0;
        // int tr_to_center_x = 0;
        // int tr_to_center_y = 0;
        // int br_to_center_x = 0;
        // int br_to_center_y = 0;
        // int bl_to_center_x = 0;
        // int bl_to_center_y = 0;

        if (key_point_.size() == 1) {
          bb_width = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2]) - center_w) * 2;
          bb_height = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1]) - center_h) * 2;

          // tl_to_center_x = bb_width / 2;
          // tl_to_center_y = bb_height / 2;
          // tr_to_center_x = bb_width / 2;
          // tr_to_center_y = bb_height / 2; 
          // br_to_center_x = bb_width / 2;
          // br_to_center_y = bb_height / 2;
          // bl_to_center_x = bb_width / 2;
          // bl_to_center_y = bb_height / 2;

        } else if (key_point_.size() == 2) { // key_point[0] is the top left, key_point[1] is bottom right
          bb_width = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2] - 
            coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2]));
          bb_height = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1] -
            coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 1]));

          // tl_to_center_x = abs(center_w - coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 0]); // bb_width / 2;
          // tl_to_center_y = abs(center_h - coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1]); //bb_height / 2;

          // tr_to_center_x = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 0] - center_w); // bb_width / 2;
          // tr_to_center_y = tl_to_center_y; // bb_height / 2; 

          // br_to_center_x = tr_to_center_x; // bb_width / 2;
          // br_to_center_y = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 1] - center_h); // bb_height / 2;

          // bl_to_center_x = tl_to_center_x; // bb_width / 2;
          // bl_to_center_y = br_to_center_y; // bb_height / 2;

        } else if (key_point_.size() == 4) {
          Dtype line01 = Distance(coords[key_point_[0] * 2], coords[key_point_[0] * 2 + 1], coords[key_point_[1] * 2], coords[key_point_[1] * 2 + 1]);
          Dtype line12 = Distance(coords[key_point_[1] * 2], coords[key_point_[1] * 2 + 1], coords[key_point_[2] * 2], coords[key_point_[2] * 2 + 1]);
          Dtype line23 = Distance(coords[key_point_[2] * 2], coords[key_point_[2] * 2 + 1], coords[key_point_[3] * 2], coords[key_point_[3] * 2 + 1]);
          Dtype line30 = Distance(coords[key_point_[3] * 2], coords[key_point_[3] * 2 + 1], coords[key_point_[0] * 2], coords[key_point_[0] * 2 + 1]);
          bb_width = (line01 + line23) / 2;
          bb_height = (line12 + line30) / 2;
          // bb_width = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2]) - center_w) * 2;
          // bb_height = abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1]) - center_h) * 2;
          // bb_width = max(bb_width, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2]) - center_w) * 2);
          // bb_height = max(bb_height, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 1]) - center_h) * 2);
          // bb_width = max(bb_width, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2]) - center_w) * 2);
          // bb_height = max(bb_height, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2 + 1]) - center_h) * 2);
          // bb_width = max(bb_width, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2]) - center_w) * 2);
          // bb_height = max(bb_height, abs(int(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2 + 1]) - center_h) * 2);

          // tl_to_center_x = abs(center_w - coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 0]); // bb_width / 2;
          // tl_to_center_y = abs(center_h - coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1]); //bb_height / 2;
          // tr_to_center_x = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 0] - center_w); // bb_width / 2;
          // tr_to_center_y = abs(center_h - coords[bid * num_anno_points_per_instance_ * 2 + key_point1[0] * 2 + 1]);; // bb_height / 2; 
          // br_to_center_x = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2 + 0] - center_w); // bb_width / 2;
          // br_to_center_y = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2 + 1] - center_h); // bb_height / 2;
          // bl_to_center_x = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2 + 0] - center_w); // bb_width / 2;
          // bl_to_center_y = abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2 + 1] - center_h); // bb_height / 2;
        }
        for (int scale_id = 0; scale_id < scale_base_.size(); scale_id ++) {
          float norm_rate = 0;
          switch (bbox_size_norm_type_) {
          case DenseboxDataParameter_BBoxSizeNormType_HEIGHT:
            norm_rate = bbox_height_ / bb_height;
            break;
          case DenseboxDataParameter_BBoxSizeNormType_WIDTH:
            norm_rate = bbox_width_ / bb_width;
            break;
          case DenseboxDataParameter_BBoxSizeNormType_DIAG:
            norm_rate = sqrt(bbox_height_ * bbox_height_ + bbox_width_ * bbox_width_) /
              sqrt(bb_width * bb_width + bb_height * bb_height);
            break;
          default:
            LOG(ERROR) << "Unknown box norm type.";
          }
          norm_rate /= scale_base_[scale_id]; // if we get a small character, the norm rate will be small, but will be bigger at lower level

          if (norm_rate < scale_ignore_lower_bounder_ || norm_rate > scale_ignore_upper_bounder_) {
            continue;
          }
          bool ignore_bbox = false;
          if (norm_rate < scale_positive_lower_bounder_ || norm_rate > scale_positive_upper_bounder_ ||
              coords[bid * num_anno_points_per_instance_ * 2 + sample_type_point_ * 2] == 2) {
            ignore_bbox = true;
          }
          // const int radius_valid_label = max(min_output_pos_radius_, int(min(bb_width, bb_height) * bbox_valid_dist_ratio_));
          const int radius_valid_label = max(min_output_pos_radius_, int(downsample_ratio * sqrt(bb_width * bb_width +  bb_height * bb_height) * bbox_valid_dist_ratio_ / 2));
          const int radius_ignore_label = radius_valid_label + ignore_margin_;
          for (int dy = -radius_ignore_label; dy <= radius_ignore_label; dy ++) {
            for (int dx = -radius_ignore_label; dx <= radius_ignore_label; dx ++) {
              float dis2center = sqrt(dx * dx + dy * dy);
              int w_label = max(min(int(dx + center_w_label), out_width_ - 1), 0);
              int h_label = max(min(int(dy + center_h_label), out_height_ - 1), 0);
              if (dis2center <= radius_valid_label && !ignore_bbox) { 
                // detection
                if (need_detection_) {
                  const int detection_label_index = ((scale_id * label_channel_per_scale + detection_channel_offset_) * out_height_ + h_label) * out_width_ + w_label;
                  top_label[detection_label_index] = 2;
                }
                // recognition
                if (need_recognition_) {
                  const int recognition_label_index = ((scale_id * label_channel_per_scale + recognition_channel_offset_)
                    * out_height_ + h_label) * out_width_ + w_label;
                  top_label[recognition_label_index] = coords[bid * num_anno_points_per_instance_ * 2 + class_index_point_ * 2];
                }
                // rotation
                if (need_rotation_) {
              	  const int rotation_label_index = ((scale_id * label_channel_per_scale + rotation_channel_offset_) 
              	    * out_height_ + h_label) * out_width_ + w_label;
              	  top_label[rotation_label_index] = coords[bid * num_anno_points_per_instance_ * 2 + rotation_point_ * 2];
                }
                // regression
                if (need_regression_) {
                  const int regression_label_index = ((scale_id * label_channel_per_scale + regression_channel_offset_) 
                    * out_height_ + h_label) * out_width_ + w_label;              
                  if (key_point_.size() == 1) {
                    top_label[regression_label_index] = bb_width * downsample_ratio / 2 - dx; // right
                    top_label[regression_label_index + 1 * out_width_ * out_height_] = bb_height * downsample_ratio / 2 - dy; // bottom
                  } else if (key_point_.size() == 2) {
                    // top_label[regression_label_index] = -bb_width * downsample_ratio / 2 - dx; // left
                    // top_label[regression_label_index + 1 * out_width_ * out_height_] = -bb_height * downsample_ratio / 2 - dy; // top
                    // top_label[regression_label_index + 2 * out_width_ * out_height_] = bb_width * downsample_ratio / 2 - dx; // right
                    // top_label[regression_label_index + 3 * out_width_ * out_height_] = bb_height * downsample_ratio / 2 - dy; // bottom
                    top_label[regression_label_index] =
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2] - center_w) * downsample_ratio - dx; // left 
                    top_label[regression_label_index + 1 * out_width_ * out_height_] =
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1] - center_h) * downsample_ratio - dy; // top
                    top_label[regression_label_index + 2 * out_width_ * out_height_] =
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2] - center_w) * downsample_ratio - dx; // right 
                    top_label[regression_label_index + 3 * out_width_ * out_height_] =
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 1] - center_h) * downsample_ratio - dy; // top

                  } else if (key_point_.size() == 4) {
                    top_label[regression_label_index] = 
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2] - center_w) * downsample_ratio - dx; // left 
                    top_label[regression_label_index + 1 * out_width_ * out_height_] = 
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[0] * 2 + 1] - center_h) * downsample_ratio - dy; // top
                    top_label[regression_label_index + 2 * out_width_ * out_height_] = 
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2] - center_w) * downsample_ratio - dx; // right 
                    top_label[regression_label_index + 3 * out_width_ * out_height_] = 
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[1] * 2 + 1] - center_h) * downsample_ratio - dy; // top
                    top_label[regression_label_index + 4 * out_width_ * out_height_] = 
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2] - center_w) * downsample_ratio - dx; // right
                    top_label[regression_label_index + 5 * out_width_ * out_height_] = 
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[2] * 2 + 1] - center_h) * downsample_ratio - dy; // bottom
                    top_label[regression_label_index + 6  * out_width_ * out_height_] = 
                      -abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2] - center_w) * downsample_ratio - dx; // left
                    top_label[regression_label_index + 7 * out_width_ * out_height_] = 
                      abs(coords[bid * num_anno_points_per_instance_ * 2 + key_point_[3] * 2 + 1] - center_h) * downsample_ratio - dy; // bottom
                  }
                }
              } else if (dis2center <= radius_ignore_label || ignore_bbox) {
                // ignore
                if (need_ignore_) {
                  // detection
                  if (need_detection_) {
                    const int detection_label_index = ((scale_id * label_channel_per_scale + detection_channel_offset_) * out_height_ + h_label) * out_width_ + w_label;
                    top_label[detection_label_index] = 1;
                  }
                  // recognition
                  if (need_recognition_) {
                    const int recognition_label_index = ((scale_id * label_channel_per_scale + recognition_channel_offset_)
                      * out_height_ + h_label) * out_width_ + w_label;
                    top_label[recognition_label_index] = 1;
                  }
                }
              }
            }
          }
        }
      
      }
    }
    top_data += input_height_ * input_width_ * img_channel;
    top_label += out_height_ * out_width_ * label_channel_;
  }
}

template <typename Dtype>
void DenseboxDataLayer<Dtype>::SetTopDataAndLabel(Dtype* top_data, Dtype* top_label, 
  vector<std::pair<float, float> >& sample_scales, vector<std::pair<std::string, vector<Dtype> > >& samples) {
  SetSampleScales(sample_scales, samples);
  vector<cv::Mat> cv_imgs;
  // std::cout << "1111111111" << std::endl;
  ScaleImagesAndCoords(sample_scales, samples, cv_imgs);
  // std::cout << "2222222222" << std::endl;
  SetCropAndPad(cv_imgs, samples);
  // std::cout << "3333333333" << std::endl;
  SetTopSamples(top_data, top_label, samples, cv_imgs);
  // std::cout << "4444444444" << std::endl;
}


// This function is called on prefetch thread
template <typename Dtype>
void DenseboxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  /// load data to fill top[0]
  // random choose images (not good enough to select only one ROI with the sample prob for each image, should be weighted)
  if (pos_sample_id_ >= pos_samples_.size() && pos_samples_.size() > 0) {
    pos_sample_id_ = 0;
    if (shuffle_)  ShuffleSamples(pos_samples_); // ShufflePosImages();
  }

  if (neg_sample_id_ >= neg_samples_.size() && neg_samples_.size() > 0) {
    neg_sample_id_ = 0;
    if (shuffle_)  ShuffleSamples(neg_samples_); // ShuffleNegImages();
  }
  // load batch sources
  vector<std::pair<std::string, vector<Dtype> > > tmp_samples;
  for (int i = 0; i < pos_batch_size_; i ++) {
    std::pair<std::string, vector<Dtype> > tmp_sample = pos_samples_[pos_sample_id_ % pos_samples_.size()];
    ShuffleGTs(tmp_sample);
    tmp_samples.push_back(tmp_sample);
    pos_sample_id_ ++;
  }

  // load negative sources
  for (int i = 0; i < neg_batch_size_; i ++) {
    tmp_samples.push_back(neg_samples_[neg_sample_id_ % neg_samples_.size()]);
    neg_sample_id_ ++;
  }

  // std::cout << "pos_batch_size_:" << pos_batch_size_ << ", neg_batch_size_:" << neg_batch_size_ << std::endl;


  ShuffleSamples(tmp_samples);  

  // distort source images and coordinates 
  vector<std::pair<float, float> > sample_scales;
  chosen_scale_index_ = caffe_rng_rand() % scale_base_.size();
  for (int i = 0; i < batch_size_; i ++) {
    sample_scales.push_back(make_pair(scale_base_[chosen_scale_index_], scale_base_[chosen_scale_index_]));
  }
  // std::cout << "top data size: " << batch->data_.num() << ", " << batch->data_.channels() << ", " << batch->data_.height() << ", " << batch->data_.width() << std::endl;
  // std::cout << "top label size: " << batch->label_.num() << ", " << batch->label_.channels() << ", " << batch->label_.height() << ", " << batch->label_.width() << std::endl;

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  SetTopDataAndLabel(top_data, top_label, sample_scales, tmp_samples);
 
  // FILE* fp1 = fopen("data.txt", "w");
  // FILE* fp2 = fopen("label.txt", "w");
  // for (int i = 0; i < batch_size_; i ++) {
  //   for (int c = 0; c < 3; c ++) {
  //     for (int h = 0; h < input_height_; h ++) {
  //       for (int w = 0; w < input_width_; w ++) {
  //         fprintf(fp1, "%.2f ", top_data[(i * 3 + c) * input_height_ * input_width_ + h * input_width_ + w]);   
  //       }
  //     }
  //   }
  //   fprintf(fp1, "\n");
  //   for (int c = 0; c < label_channel_; c ++) {
  //     for (int h = 0; h < out_height_; h ++) {
  //       for (int w = 0; w < out_width_; w ++) {
  //         fprintf(fp2, "%.2f ", top_label[(i * label_channel_ + c) * out_height_ * out_width_ + h * out_width_ + w]);  
  //       }
  //     }
  //   }
  //   fprintf(fp2, "\n");
  // }
  // fclose(fp1);
  // fclose(fp2);
  // std::cout << "Written ..." << std::endl;
  // CHECK_EQ(1, 3) << "Done" << tmp_samples[0].first;
}

INSTANTIATE_CLASS(DenseboxDataLayer);
REGISTER_LAYER_CLASS(DenseboxData);

}  // namespace caffe
#endif  // USE_OPENCV
