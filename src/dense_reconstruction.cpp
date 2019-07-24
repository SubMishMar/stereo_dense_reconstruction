#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "elas.h"

using namespace cv;
using namespace std;

Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy;
Vec3d T;
FileStorage calib_file;
int debug = 0;
Size out_img_size;
Size calib_img_size;

image_transport::Publisher dmap_pub;
ros::Publisher pcl_pub;

void publishPointCloud(Mat& img_left, Mat& dmap) {

  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  vector< Point3d > points;
  sensor_msgs::PointCloud pc;
  sensor_msgs::ChannelFloat32 ch;
  ch.name = "rgb";
  pc.header.frame_id = "jackal";
  pc.header.stamp = ros::Time::now();
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      int d = dmap.at<uchar>(j,i);
      // if low disparity, then ignore
      if (d < 2) {
        continue;
      }
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      pos = Q * V; // 3D homogeneous coordinate
      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;
      points.push_back(Point3d(point3d_robot));
      geometry_msgs::Point32 pt;
      pt.x = point3d_robot.at<double>(0,0);
      pt.y = point3d_robot.at<double>(1,0);
      pt.z = point3d_robot.at<double>(2,0);
      pc.points.push_back(pt);
      int32_t red, blue, green;
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];
      int32_t rgb = (red << 16 | green << 8 | blue);
      ch.values.push_back(*reinterpret_cast<float*>(&rgb));
    }
  }
  if (!dmap.empty()) {
    sensor_msgs::ImagePtr disp_msg;
    disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    dmap_pub.publish(disp_msg);
  }
  pc.channels.push_back(ch);
  pcl_pub.publish(pc);
}

Mat generateDisparityMap(Mat& left, Mat& right) {
  if (left.empty() || right.empty()) 
    return left;
  const Size imsize = left.size();
  const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  Elas::parameters param(Elas::MIDDLEBURY);
  param.postprocess_only_left = true;
  Elas elas(param);
  elas.process(left.data, right.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
  Mat dmap = Mat(out_img_size, CV_8UC1, Scalar(0));
  leftdpf.convertTo(dmap, CV_8U, 1.);
  return dmap;
}

void imgCallback(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right) {
  Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
  Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;
  if (tmpL.empty() || tmpR.empty())
    return;
  
  Mat img_left, img_right, img_left_color;
  remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR);
  remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  cvtColor(img_left, img_left_color, CV_GRAY2BGR);
  
  Mat dmap = generateDisparityMap(img_left, img_right);
  publishPointCloud(img_left_color, dmap);
  
  imshow("LEFT", img_left);
  imshow("RIGHT", img_right);
  imshow("DISP", dmap);
  waitKey(30);
}

void findRectificationMap(FileStorage& calib_file, Size finalSize) {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, 
                CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, finalSize, CV_32F, rmapx, rmapy);
  cout << "Done rectification" << endl;
}

template <typename T>
T readParam(ros::NodeHandle &n, std::string name){
    T ans;
    if (n.getParam(name, ans)){
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else{
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "stereo_dense_reconstruction");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  
  std::string left_img_topic =
          readParam<std::string>(nh, "left_img_topic");
  std::string right_img_topic =
            readParam<std::string>(nh, "right_img_topic");
  std::string calib_file_name =
            readParam<std::string>(nh, "calib_file_name");

  int calib_width, calib_height, out_width, out_height;

  calib_img_size = Size(calib_width, calib_height);
  out_img_size = Size(out_width, out_height);
  
  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;
  
  findRectificationMap(calib_file, out_img_size);
  
  message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, left_img_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, right_img_topic, 1);
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
  sync.registerCallback(boost::bind(&imgCallback, _1, _2));

  dmap_pub = it.advertise("/camera/left/disparity_map", 1);
  pcl_pub = nh.advertise<sensor_msgs::PointCloud>("/camera/left/point_cloud",1);

  ros::spin();
  return 0;
}