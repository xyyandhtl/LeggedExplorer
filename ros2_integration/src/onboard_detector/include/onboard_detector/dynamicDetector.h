/*
    FILE: dynamicDetector.h
    ---------------------------------
    header file of dynamic obstacle detector
*/
#ifndef ONBOARDDETECTOR_DYNAMICDETECTOR_H
#define ONBOARDDETECTOR_DYNAMICDETECTOR_H

#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Eigen>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <onboard_detector/srv/get_dynamic_obstacles.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <onboard_detector/detectors/dbscan.h>
#include <onboard_detector/detectors/uvDetector.h>
#include <onboard_detector/tracking/kalmanFilter.h>
#include <onboard_detector/utils.h>

using std::cout; using std::endl;
namespace onboardDetector{
    class dynamicDetector : public rclcpp::Node{
    private:
        std::string ns_;
        std::string hint_;

        // ROS
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, geometry_msgs::msg::PoseStamped> depthPoseSync;
        std::shared_ptr<message_filters::Synchronizer<depthPoseSync>> depthPoseSync_;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, nav_msgs::msg::Odometry> depthOdomSync;
        std::shared_ptr<message_filters::Synchronizer<depthOdomSync>> depthOdomSync_;        

        // Subscriber
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depthSub_;
        std::shared_ptr<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>> poseSub_;
        std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odomSub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr colorImageSub_;
        rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yoloDetectionSub_;


        // Publisher
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr uvDepthMapPub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr uDepthMapPub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr uvBirdViewPub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detectedColorImgPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr uvBBoxesPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dbBBoxesPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filteredBBoxesPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackedBBoxesPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dynamicBBoxesPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr historyTrajPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr velVisPub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filteredPointsPub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamicPointsPub_;

        // Timer
        rclcpp::TimerBase::SharedPtr detectionTimer_;
        rclcpp::TimerBase::SharedPtr trackingTimer_;
        rclcpp::TimerBase::SharedPtr classificationTimer_;
        rclcpp::TimerBase::SharedPtr visTimer_;

        //Server
        rclcpp::Service<onboard_detector::srv::GetDynamicObstacles>::SharedPtr getDynamicObstacleServer_;


        // ---------------PARAMETERS----------------
        // TOPICS
        int localizationMode_;
        std::string depthTopicName_;
        std::string colorImgTopicName_;
        std::string poseTopicName_;
        std::string odomTopicName_;

        // CAMERA (depth and color)
        double fx_, fy_, cx_, cy_; // depth camera intrinsics
        double fxC_, fyC_, cxC_, cyC_;
        double depthScale_; // value / depthScale
        double depthMinValue_, depthMaxValue_;
        int depthFilterMargin_, skipPixel_; // depth filter margin
        int imgCols_, imgRows_;
        Eigen::Matrix4d body2CamDepth_; // from body frame to depth camera frame
        Eigen::Matrix4d body2CamColor_; // from body frame to color camera frame

        // DETECTOR PARAM
        double dt_;
        double raycastMaxLength_;
        double boxIOUThresh_;

        // DBSCAN
        double groundHeight_;
        int dbMinPointsCluster_;
        double dbEpsilon_;
        int voxelOccThresh_;

        // TRACKING
        int histSize_;
        int predSize_;
        double simThresh_;
        double simThreshRetrack_;
        int fixSizeHistThresh_;
        double fixSizeDimThresh_;
        double eP_; // kalman filter initial uncertainty matrix
        double eQPos_; // motion model uncertainty matrix for position
        double eQVel_; // motion model uncertainty matrix for velocity
        double eQAcc_; // motion model uncertainty matrix for acceleration
        double eRPos_; // observation uncertainty matrix for position
        double eRVel_; // observation uncertainty matrix for velocity
        double eRAcc_; // observation uncertainty matrix for acceleration
        int kfAvgFrames_;

        // CLASSIFICATION
        int skipFrame_;
        double dynaVelThresh_;
        double dynaVoteThresh_;
        double maxSkipRatio_;
        int forceDynaFrames_;
        int forceDynaCheckRange_;
        int dynamicConsistThresh_;
        
        // SIZE CONSTRAINT
        bool constrainSize_;
        std::vector<Eigen::Vector3d> targetObjectSize_; 
        // ---------------------------------------------

        // --------------DETECTOR DATA------------------
        // DETECTOR
        std::shared_ptr<onboardDetector::UVdetector> uvDetector_;
        std::shared_ptr<onboardDetector::DBSCAN> dbCluster_;

        // SENSOR DATA
        cv::Mat depthImage_;
        Eigen::Vector3d position_; // robot camera position
        Eigen::Matrix3d orientation_; // robot orientation
        Eigen::Vector3d positionDepth_; // depth camera position
        Eigen::Matrix3d orientationDepth_; // depth camera orientation
        Eigen::Vector3d positionColor_; // color camera position
        Eigen::Matrix3d orientationColor_; // color camera orientation
        Eigen::Vector3d localSensorRange_ {5.0, 5.0, 5.0};
        
        // DBSCAN DATA
        int projPointsNum_ = 0;
        std::vector<Eigen::Vector3d> projPoints_; // projected points from depth image
        std::vector<double> pointsDepth_; 
        std::vector<Eigen::Vector3d> filteredPoints_; // filtered point cloud data for DBSCAN
        std::vector<onboardDetector::box3D> dbBBoxes_; // DBSCAN bounding boxes        
        std::vector<std::vector<Eigen::Vector3d>> pcClusters_; // pointcloud clusters
        std::vector<Eigen::Vector3d> pcClusterCenters_; // pointcloud cluster centers
        std::vector<Eigen::Vector3d> pcClusterStds_; // pointcloud cluster standard deviation in each axis

        // U-DEPTH DETECTOR DATA
        std::vector<onboardDetector::box3D> uvBBoxes_; // uv detector bounding boxes


        // DETECTOR DATA
        std::vector<onboardDetector::box3D> filteredBBoxes_; // filtered bboxes
        std::vector<std::vector<Eigen::Vector3d>> filteredPcClusters_; // pointcloud clusters after filtering by UV and DBSCAN fusion
        std::vector<Eigen::Vector3d> filteredPcClusterCenters_; // filtered pointcloud cluster centers
        std::vector<Eigen::Vector3d> filteredPcClusterStds_; // filtered pointcloud cluster standard deviation in each axis


        // TRACKING AND ASSOCIATION DATA
        std::vector<onboardDetector::kalman_filter> filters_; // kalman filter for each objects
        bool newDetectFlag_;
        std::deque<Eigen::Vector3d> positionHist_; // current position
		std::deque<Eigen::Matrix3d> orientationHist_; // current orientation
        std::vector<std::deque<onboardDetector::box3D>> boxHist_; // data association result: history of filtered bounding boxes for each box in current frame
        std::vector<std::deque<std::vector<Eigen::Vector3d>>> pcHist_; // data association result: history of filtered pc clusteres for each pc cluster in current frame
        std::vector<onboardDetector::box3D> trackedBBoxes_; // bboxes tracked from kalman filtering
        std::vector<onboardDetector::box3D> dynamicBBoxes_; // boxes classified as dynamic

        // YOLO DATA
        vision_msgs::msg::Detection2DArray yoloDetectionResults_; // yolo detected 2D results
        cv::Mat detectedColorImage_;

    
    public:
        dynamicDetector();
        void initParam();
        void registerPub();
        void registerCallback();

        // service
		bool getDynamicObstaclesSrv(const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Request> req, 
								    const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Response> res);

        // callback
        void depthPoseCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose);
        void depthOdomCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const nav_msgs::msg::Odometry::ConstSharedPtr& odom);
        void colorImgCB(const sensor_msgs::msg::Image::ConstSharedPtr& img);
        void yoloDetectionCB(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections);
        void detectionCB();
        void trackingCB();
        void classificationCB();
        void visCB();

        // detection function
        void dbscanDetect();
        void uvDetect();
        void filterBBoxes();

        // DBSCAN detector function
        void projectDepthImage();
        void filterPoints(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints);
        void clusterPointsAndBBoxes(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::box3D>& bboxes, std::vector<std::vector<Eigen::Vector3d>>& pcClusters, std::vector<Eigen::Vector3d>& pcClusterCenters, std::vector<Eigen::Vector3d>& pcClusterStds);
        void voxelFilter(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints);
        void calcPcFeat(const std::vector<Eigen::Vector3d>& pcCluster, Eigen::Vector3d& pcClusterCenter, Eigen::Vector3d& pcClusterStd);

        // uv Detector function
        void transformUVBBoxes(std::vector<onboardDetector::box3D>& bboxes);

        // data association and tracking functions
        void boxAssociation(std::vector<int>& bestMatch, std::vector<int> &boxOOR);
        void boxAssociationHelper(std::vector<int>& bestMatch, std::vector<int> &boxOOR);
        void linearProp(std::vector<onboardDetector::box3D>& propedBoxes);
        void genFeat(const std::vector<onboardDetector::box3D>& propedBoxes, int numObjs, std::vector<Eigen::VectorXd>& propedBoxesFeat, std::vector<Eigen::VectorXd>& currBoxesFeat);
        void genFeatHelper(std::vector<Eigen::VectorXd>& feature, const std::vector<onboardDetector::box3D>& boxes);
        void findBestMatch(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch);
        void findBestMatchEstimate(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch, std::vector<int>& boxOOR);
        void getBoxOutofRange(std::vector<int>& boxOOR, const std::vector<int>&bestMatch); 
        int getEstimateFrameNum(const std::deque<onboardDetector::box3D> &boxHist);
        void kalmanFilterAndUpdateHist(const std::vector<int>& bestMatch, const std::vector<int> &boxOOR);
        void kalmanFilterMatrixAcc(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R);
        void getKalmanObservationAcc(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z);
        void getEstimateBox(const std::deque<onboardDetector::box3D> &boxHist, onboardDetector::box3D &estimatedBBox);

        // user functions
        void getDynamicObstacles(std::vector<onboardDetector::box3D>& incomeDynamicBBoxes, const Eigen::Vector3d &robotSize = Eigen::Vector3d(0.0,0.0,0.0));
        void getDynamicObstaclesHist(std::vector<std::vector<Eigen::Vector3d>>& posHist, 
									 std::vector<std::vector<Eigen::Vector3d>>& velHist, 
									 std::vector<std::vector<Eigen::Vector3d>>& sizeHist, const Eigen::Vector3d &robotSize = Eigen::Vector3d(0.0,0.0,0.0));

        // visualization
        void getDynamicPc(std::vector<Eigen::Vector3d>& dynamicPc);
        void publishUVImages(); 
        void publishColorImages();
        void publishPoints(const std::vector<Eigen::Vector3d>& points, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher);
        void publish3dBox(const std::vector<onboardDetector::box3D>& bboxes, const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher, double r, double g, double b);
        void publishHistoryTraj();
        void publishVelVis();

        // helper fucntion       
        void updatePoseHist();
        void transformBBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size, const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation,
                                  Eigen::Vector3d& newCenter, Eigen::Vector3d& newSize);
        bool isInFov(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, Eigen::Vector3d& point);
        int getBestOverlapBBox(const onboardDetector::box3D& currBBox, const std::vector<onboardDetector::box3D>& targetBBoxes, double& bestIOU);
        double calBoxIOU(const onboardDetector::box3D& box1, const onboardDetector::box3D& box2);

        // inline helper function
        bool isInFilterRange(const Eigen::Vector3d& pos);
        void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx, double res);
        int indexToAddress(const Eigen::Vector3i& idx, double res);
        int posToAddress(const Eigen::Vector3d& pos, double res);
        void indexToPos(const Eigen::Vector3i& idx, Eigen::Vector3d& pos, double res);
        void getSensorPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose, const Eigen::Matrix4d& sensorTransform, Eigen::Matrix4d& camPoseMatrix);
        void getSensorPose(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const Eigen::Matrix4d& sensorTransform, Eigen::Matrix4d& camPoseMatrix);
        onboardDetector::Point eigenToDBPoint(const Eigen::Vector3d& p);
        Eigen::Vector3d dbPointToEigen(const onboardDetector::Point& pDB);
        void eigenToDBPointVec(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::Point>& pointsDB, int size);
    };

    inline bool dynamicDetector::isInFilterRange(const Eigen::Vector3d& pos){
        if ((pos(0) >= this->position_(0) - this->localSensorRange_(0)) and (pos(0) <= this->position_(0) + this->localSensorRange_(0)) and 
            (pos(1) >= this->position_(1) - this->localSensorRange_(1)) and (pos(1) <= this->position_(1) + this->localSensorRange_(1)) and 
            (pos(2) >= this->position_(2) - this->localSensorRange_(2)) and (pos(2) <= this->position_(2) + this->localSensorRange_(2))){
            return true;
        }
        else{
            return false;
        }        
    }

    inline void dynamicDetector::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx, double res){
        idx(0) = floor( (pos(0) - this->position_(0) + localSensorRange_(0)) / res);
        idx(1) = floor( (pos(1) - this->position_(1) + localSensorRange_(1)) / res);
        idx(2) = floor( (pos(2) - this->position_(2) + localSensorRange_(2)) / res);
    }

    inline int dynamicDetector::indexToAddress(const Eigen::Vector3i& idx, double res){
        return idx(0) * ceil(2*this->localSensorRange_(1)/res) * ceil(2*this->localSensorRange_(2)/res) + idx(1) * ceil(2*this->localSensorRange_(2)/res) + idx(2);
    }

    inline int dynamicDetector::posToAddress(const Eigen::Vector3d& pos, double res){
        Eigen::Vector3i idx;
        this->posToIndex(pos, idx, res);
        return this->indexToAddress(idx, res);
    }

    inline void dynamicDetector::indexToPos(const Eigen::Vector3i& idx, Eigen::Vector3d& pos, double res){
		pos(0) = (idx(0) + 0.5) * res - localSensorRange_(0) + this->position_(0);
		pos(1) = (idx(1) + 0.5) * res - localSensorRange_(1) + this->position_(1);
		pos(2) = (idx(2) + 0.5) * res - localSensorRange_(2) + this->position_(2);
	}

    inline void dynamicDetector::getSensorPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose, const Eigen::Matrix4d& sensorTransform, Eigen::Matrix4d& camPoseMatrix){
      Eigen::Quaterniond quat;
      quat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
      Eigen::Matrix3d rot = quat.toRotationMatrix();

      // convert body pose to camera pose
      Eigen::Matrix4d map2body; map2body.setZero();
      map2body.block<3, 3>(0, 0) = rot;
      map2body(0, 3) = pose->pose.position.x; 
      map2body(1, 3) = pose->pose.position.y;
      map2body(2, 3) = pose->pose.position.z;
      map2body(3, 3) = 1.0;

      camPoseMatrix = map2body * sensorTransform;
    }

    inline void dynamicDetector::getSensorPose(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const Eigen::Matrix4d& sensorTransform, Eigen::Matrix4d& camPoseMatrix){
      Eigen::Quaterniond quat;
      quat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
      Eigen::Matrix3d rot = quat.toRotationMatrix();

      // convert body pose to camera pose
      Eigen::Matrix4d map2body; map2body.setZero();
      map2body.block<3, 3>(0, 0) = rot;
      map2body(0, 3) = odom->pose.pose.position.x; 
      map2body(1, 3) = odom->pose.pose.position.y;
      map2body(2, 3) = odom->pose.pose.position.z;
      map2body(3, 3) = 1.0;

      camPoseMatrix = map2body * sensorTransform;
    }

    inline onboardDetector::Point dynamicDetector::eigenToDBPoint(const Eigen::Vector3d& p){
        onboardDetector::Point pDB;
        pDB.x = p(0);
        pDB.y = p(1);
        pDB.z = p(2);
        pDB.clusterID = -1;
        return pDB;
    }

    inline Eigen::Vector3d dynamicDetector::dbPointToEigen(const onboardDetector::Point& pDB){
        Eigen::Vector3d p;
        p(0) = pDB.x;
        p(1) = pDB.y;
        p(2) = pDB.z;
        return p;
    }

    inline void dynamicDetector::eigenToDBPointVec(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::Point>& pointsDB, int size){
        for (int i=0; i<size; ++i){
            Eigen::Vector3d p = points[i];
            onboardDetector::Point pDB = this->eigenToDBPoint(p);
            pointsDB.push_back(pDB);
        }
    }
}
#endif