#include <ros/ros.h>

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
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
  pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  vector< Point3d > points;

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
      int32_t red, blue, green;
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];

      pcl::PointXYZRGB pt(red, green, blue);

      pt.x = Z;
      pt.y = -X;
      pt.z = -Y;
      out_cloud_pcl.push_back(pt);
    }
  }
  if (!dmap.empty()) {
    sensor_msgs::ImagePtr disp_msg;
    disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    dmap_pub.publish(disp_msg);
  }
  sensor_msgs::PointCloud2 out_cloud_ros;
  pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
  out_cloud_ros.header.frame_id = "left_camera";
  out_cloud_ros.header.stamp = ros::Time::now();
  pcl_pub.publish(out_cloud_ros);
}

Mat generateDisparityMapELAS(Mat& left, Mat& right) {
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
  Mat tmpL_color = cv_bridge::toCvShare(msg_left, sensor_msgs::image_encodings::BGR8)->image;
  Mat tmpR_color = cv_bridge::toCvShare(msg_right, sensor_msgs::image_encodings::BGR8)->image;

  if (tmpL_color.empty() || tmpR_color.empty())
    return;

  cv::Mat tmpL, tmpR;
  cvtColor(tmpL_color, tmpL, CV_BGR2GRAY);
  cvtColor(tmpR_color, tmpR, CV_BGR2GRAY);
  Mat img_left, img_right, img_left_color;
  remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR);
  remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);

  Mat dmap = generateDisparityMapELAS(img_left, img_right);
  publishPointCloud(tmpL_color, dmap);
  
  imshow("LEFT", tmpL_color);
  imshow("RIGHT", tmpR_color);
  imshow("DISP", dmap);
  waitKey(30);
}

void findRectificationMap() {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, 
                CALIB_ZERO_DISPARITY, 0, calib_img_size, &validRoi[0], &validRoi[1]);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, calib_img_size, CV_32F, lmapx, lmapy);

  cv::initUndistortRectifyMap(K2, D2, R2, P2, calib_img_size, CV_32F, rmapx, rmapy);

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

  calib_width =
          readParam<int>(nh, "calib_width");
  calib_height =
          readParam<int>(nh, "calib_height");

  calib_img_size = Size(calib_width, calib_height);

  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;

  findRectificationMap();
  
  message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, left_img_topic, 100);
  message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, right_img_topic, 100);
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
  sync.registerCallback(boost::bind(&imgCallback, _1, _2));

  dmap_pub = it.advertise("/camera/left/disparity_map", 1);
  pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/left/point_cloud",1);

  ros::spin();
  return 0;
}