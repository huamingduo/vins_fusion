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

#include "feature_tracker.h"

std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img,
                                                                                                   const cv::Mat &_img1) {
  cur_time = _cur_time;
  cur_img = _img;
  row = cur_img.rows;
  col = cur_img.cols;
  cv::Mat rightImg = _img1;

  cur_pts.clear();
  if (prev_pts.size() > 0) {
    TicToc t_o;
    std::vector<uchar> status;
    std::vector<float> err;
    if (hasPrediction) {
      cur_pts = predict_pts;
      cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

      int succ_num = 0;
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) succ_num++;
      }
      if (succ_num < 10) {
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
      }
    } else
      cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);

    // reverse check
    if (FLOW_BACK) {
      vector<uchar> reverse_status;
      vector<cv::Point2f> reverse_pts = prev_pts;
      cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
      // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
          status[i] = 1;
        } else
          status[i] = 0;
      }
    }

    for (int i = 0; i < int(cur_pts.size()); i++) {
      if (status[i] && !inBorder(cur_pts[i])) {
        status[i] = 0;
      }
    }
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
  }

  for (auto &n : track_cnt) n++;

  if (1) {
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1) cout << "mask type wrong " << endl;
      cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
    } else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

    for (auto &p : n_pts) {
      cur_pts.push_back(p);
      ids.push_back(n_id++);
      track_cnt.push_back(1);
    }
  }

  cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
  pts_velocity = ptsVelocity(ids, cur_un_pts, prev_un_pts_map, cur_un_pts_map);

  if (!_img1.empty() && stereo_cam) {
    ids_right.clear();
    cur_right_pts.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();
    if (!cur_pts.empty()) {
      // printf("stereo image; track feature on right image\n");
      vector<cv::Point2f> reverseLeftPts;
      vector<uchar> status, statusRightLeft;
      vector<float> err;
      // cur left ---- cur right
      cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
      // reverse check cur right ---- cur left
      if (FLOW_BACK) {
        cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
            status[i] = 1;
          else
            status[i] = 0;
        }
      }

      ids_right = ids;
      reduceVector(cur_right_pts, status);
      reduceVector(ids_right, status);
      // only keep left-right pts
      /*
      reduceVector(cur_pts, status);
      reduceVector(ids, status);
      reduceVector(track_cnt, status);
      reduceVector(cur_un_pts, status);
      reduceVector(pts_velocity, status);
      */
      cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
      right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, prev_un_right_pts_map, cur_un_right_pts_map);
    }
    prev_un_right_pts_map = cur_un_right_pts_map;
  }

  if (SHOW_TRACK) {
    drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++) {
      prevLeftPtsMap[ids[i]] = cur_pts[i];
    }
  }

  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  prev_un_pts_map = cur_un_pts_map;
  prev_time = cur_time;
  hasPrediction = false;

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (size_t i = 0; i < ids.size(); i++) {
    double x = cur_un_pts[i].x;
    double y = cur_un_pts[i].y;
    double z = 1;
    double p_u = cur_pts[i].x;
    double p_v = cur_pts[i].y;
    double velocity_x = pts_velocity[i].x;
    double velocity_y = pts_velocity[i].y;
    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;

    int camera_id = 0;
    const int &feature_id = ids[i];
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  if (!_img1.empty() && stereo_cam) {
    for (size_t i = 0; i < ids_right.size(); i++) {
      double x = cur_un_right_pts[i].x;
      double y = cur_un_right_pts[i].y;
      double z = 1;
      double p_u = cur_right_pts[i].x;
      double p_v = cur_right_pts[i].y;
      double velocity_x = right_pts_velocity[i].x;
      double velocity_y = right_pts_velocity[i].y;
      Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;

      int camera_id = 1;
      const int &feature_id = ids_right[i];
      featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

  return featureFrame;
}

void FeatureTracker::setMask() {
  mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(cur_pts[i], ids[i])));
  }

  std::sort(
      cnt_pts_id.begin(), cnt_pts_id.end(),
      [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b) { return a.first > b.first; });

  cur_pts.clear();
  ids.clear();
  track_cnt.clear();
  cur_pts.reserve(cnt_pts_id.size());
  ids.reserve(cnt_pts_id.size());
  track_cnt.reserve(cnt_pts_id.size());

  for (const auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) != 255) {
      continue;
    }
    cur_pts.push_back(it.second.first);
    ids.push_back(it.second.second);
    track_cnt.push_back(it.first);
    cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
  }
  cur_pts.resize(cur_pts.size());
  ids.resize(ids.size());
  track_cnt.resize(track_cnt.size());
}

std::vector<cv::Point2f> FeatureTracker::undistortedPts(const std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam) {
  std::vector<cv::Point2f> un_pts;
  un_pts.reserve(pts.size());
  for (unsigned int i = 0; i < pts.size(); i++) {
    Eigen::Vector2d a(pts[i].x, pts[i].y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
  return un_pts;
}

std::vector<cv::Point2f> FeatureTracker::ptsVelocity(const std::vector<int> &ids, const std::vector<cv::Point2f> &pts,
                                                     const std::map<int, cv::Point2f> &prev_id_pts, std::map<int, cv::Point2f> &cur_id_pts) {
  cur_id_pts.clear();
  for (unsigned int i = 0; i < ids.size(); i++) {
    cur_id_pts.insert(std::make_pair(ids[i], pts[i]));
  }

  // caculate points velocity
  std::vector<cv::Point2f> pts_velocity;
  if (prev_id_pts.empty()) {
    pts_velocity.resize(cur_pts.size(), cv::Point2f(0, 0));
    return pts_velocity;
  }

  double dt = cur_time - prev_time;
  pts_velocity.resize(pts.size(), cv::Point2f(0, 0));
  for (unsigned int i = 0; i < pts.size(); i++) {
    const auto it = prev_id_pts.find(ids[i]);
    if (it != prev_id_pts.end()) {
      double v_x = (pts[i].x - it->second.x) / dt;
      double v_y = (pts[i].y - it->second.y) / dt;
      pts_velocity[i] = cv::Point2f(v_x, v_y);
    }
  }

  return pts_velocity;
}

void FeatureTracker::readIntrinsicParameter(const std::vector<std::string> &calib_file) {
  for (const auto &file : calib_file) {
    ROS_INFO("reading paramerter of camera %s", file.c_str());
    camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(file);
    m_camera.push_back(camera);
  }

  stereo_cam = calib_file.size() == 2;
}

void FeatureTracker::setPrediction(const std::map<int, Eigen::Vector3d> &predictPts) {
  hasPrediction = true;
  predict_pts.clear();
  predict_pts.reserve(ids.size());
  predict_pts_debug.clear();
  predict_pts_debug.reserve(ids.size());

  for (size_t i = 0; i < ids.size(); ++i) {
    const auto it = predictPts.find(ids[i]);
    if (it != predictPts.end()) {
      Eigen::Vector2d tmp_uv;
      m_camera[0]->spaceToPlane(it->second, tmp_uv);
      predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
      predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
    } else {
      predict_pts.push_back(prev_pts[i]);
    }
  }
}

void FeatureTracker::removeOutliers(const std::set<int> &ids_to_remove) {
  std::vector<uchar> status;
  status.reserve(ids.size());
  for (const auto &id : ids) {
    const auto it = ids_to_remove.find(id);
    if (it != ids_to_remove.end()) {
      status.push_back(0);
    } else {
      status.push_back(1);
    }
  }

  reduceVector(prev_pts, status);
  reduceVector(ids, status);
  reduceVector(track_cnt, status);
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts, map<int, cv::Point2f> &prevLeftPtsMap) {
  // int rows = imLeft.rows;
  int cols = imLeft.cols;
  if (!imRight.empty() && stereo_cam)
    cv::hconcat(imLeft, imRight, imTrack);
  else
    imTrack = imLeft.clone();
  cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

  for (size_t j = 0; j < curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  if (!imRight.empty() && stereo_cam) {
    for (size_t i = 0; i < curRightPts.size(); i++) {
      cv::Point2f rightPt = curRightPts[i];
      rightPt.x += cols;
      cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
      // cv::Point2f leftPt = curLeftPtsTrackRight[i];
      // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  }

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < curLeftIds.size(); i++) {
    int id = curLeftIds[i];
    mapIt = prevLeftPtsMap.find(id);
    if (mapIt != prevLeftPtsMap.end()) {
      cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }
}