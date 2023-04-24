/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <execinfo.h>

#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

class FeatureTracker {
 public:
  FeatureTracker() {}

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img,
                                                                                     const cv::Mat &_img1 = cv::Mat());
  void readIntrinsicParameter(const vector<string> &calib_file);
  void setPrediction(const std::map<int, Eigen::Vector3d> &predictPts);
  void removeOutliers(const std::set<int> &ids_to_remove);

  cv::Mat getTrackImage() { return imTrack; }

 private:
  void setMask();
  std::vector<cv::Point2f> undistortedPts(const std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
  std::vector<cv::Point2f> ptsVelocity(const std::vector<int> &ids, const std::vector<cv::Point2f> &pts,
                                       const std::map<int, cv::Point2f> &prev_id_pts, std::map<int, cv::Point2f> &cur_id_pts);

  void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                 vector<cv::Point2f> &curRightPts, map<int, cv::Point2f> &prevLeftPtsMap);

  inline bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
  }

  inline double distance(const cv::Point2f &pt1, const cv::Point2f &pt2) {
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
  }

  inline void reduceVector(std::vector<cv::Point2f> &v, const std::vector<uchar> &status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
      if (status[i]) v[j++] = v[i];
    v.resize(j);
  }

  inline void reduceVector(std::vector<int> &v, const std::vector<uchar> &status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
      if (status[i]) v[j++] = v[i];
    v.resize(j);
  }

 private:
  bool hasPrediction = false;
  bool stereo_cam = false;

  int n_id = 0;
  int row, col;

  double cur_time;
  double prev_time;

  cv::Mat imTrack;
  cv::Mat mask;
  cv::Mat prev_img, cur_img;

  vector<int> ids, ids_right;
  vector<int> track_cnt;
  vector<cv::Point2f> n_pts;
  vector<cv::Point2f> predict_pts;
  vector<cv::Point2f> predict_pts_debug;
  vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
  vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
  vector<cv::Point2f> pts_velocity, right_pts_velocity;
  vector<camodocal::CameraPtr> m_camera;

  map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
  map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
  map<int, cv::Point2f> prevLeftPtsMap;
};
