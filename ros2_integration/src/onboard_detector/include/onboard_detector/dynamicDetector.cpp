/*
    FILE: dynamicDetector.cpp
    ---------------------------------
    function implementation of dynamic osbtacle detector
*/
#include <onboard_detector/dynamicDetector.h>

namespace onboardDetector{
    dynamicDetector::dynamicDetector() : Node("dynamic_detector_node"){
        this->ns_ = "onboard_detector";
        this->hint_ = "[onboardDetector]";
        this->initParam();
        this->registerPub();
        this->registerCallback();
    }

    void dynamicDetector::initParam(){
		// localization mode
		this->declare_parameter<int>("localization_mode", 0);
		this->get_parameter("localization_mode", this->localizationMode_);
		std::cout << this->hint_ << ": Localization mode: pose (0)/odom (1). Your option: " << this->localizationMode_ << std::endl;

        // depth topic name
		this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_raw");
		this->get_parameter("depth_image_topic", this->depthTopicName_);
        cout << this->hint_ << ": Depth topic: " << this->depthTopicName_ << endl;
        
        // color topic name
		this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
		this->get_parameter("color_image_topic", this->colorImgTopicName_);
        cout << this->hint_ << ": Color image topic: " << this->colorImgTopicName_ << endl;

        if (this->localizationMode_ == 0){
            // pose topic name
            this->declare_parameter<std::string>("pose_topic", "/CERLAB/quadcopter/pose");
            this->get_parameter("pose_topic", this->poseTopicName_);
            cout << this->hint_ << ": Pose topic: " << this->poseTopicName_ << endl;
        }
        else{
            // odom topic name
            this->declare_parameter<std::string>("odom_topic", "/CERLAB/quadcopter/odom");
            this->get_parameter("odom_topic", this->odomTopicName_);
            cout << this->hint_ << ": Odom topic: " << this->odomTopicName_ << endl;
        }

        // depth intrinsics
        std::vector<double> depthIntrinsics;
        this->declare_parameter<std::vector<double>>("depth_intrinsics", {0.0, 0.0, 0.0, 0.0});
        this->get_parameter("depth_intrinsics", depthIntrinsics);
        this->fx_ = depthIntrinsics[0];
        this->fy_ = depthIntrinsics[1];
        this->cx_ = depthIntrinsics[2];
        this->cy_ = depthIntrinsics[3];
        std::cout << this->hint_ << ": fx, fy, cx, cy: [" << this->fx_ << ", " << this->fy_ << ", " << this->cx_ << ", " << this->cy_ << "]" << std::endl;
    
        // color camera intrinsics
        std::vector<double> colorIntrinsics;
        this->declare_parameter<std::vector<double>>("color_intrinsics", {0.0, 0.0, 0.0, 0.0});
        this->get_parameter("color_intrinsics", colorIntrinsics);
        this->fxC_ = colorIntrinsics[0];
        this->fyC_ = colorIntrinsics[1];
        this->cxC_ = colorIntrinsics[2];
        this->cyC_ = colorIntrinsics[3];
        std::cout << this->hint_ << ": fxC, fyC, cxC, cyC: [" << this->fxC_ << ", " << this->fyC_ << ", " << this->cxC_ << ", " << this->cyC_ << "]" << std::endl;
    
        // depth scale factor
        this->declare_parameter<double>("depth_scale_factor", 1000.0);
        this->get_parameter("depth_scale_factor", this->depthScale_);
        std::cout << this->hint_ << ": Depth scale factor: " << this->depthScale_ << std::endl;

        // depth min value
        this->declare_parameter<double>("depth_min_value", 0.2);
        this->get_parameter("depth_min_value", this->depthMinValue_);
        std::cout << this->hint_ << ": Depth min value: " << this->depthMinValue_ << std::endl;

        // depth max value
        this->declare_parameter<double>("depth_max_value", 5.0);
        this->get_parameter("depth_max_value", this->depthMaxValue_);
        std::cout << this->hint_ << ": Depth max value: " << this->depthMaxValue_ << std::endl;

        // depth filter margin
        this->declare_parameter<int>("depth_filter_margin", 0);
        this->get_parameter("depth_filter_margin", this->depthFilterMargin_);
        std::cout << this->hint_ << ": Depth filter margin: " << this->depthFilterMargin_ << std::endl;

        // depth skip pixel
        this->declare_parameter<int>("depth_skip_pixel", 1);
        this->get_parameter("depth_skip_pixel", this->skipPixel_);
        std::cout << this->hint_ << ": Depth skip pixel: " << this->skipPixel_ << std::endl;

        // depth image columns
        this->declare_parameter<int>("image_cols", 640);
        this->get_parameter("image_cols", this->imgCols_);
        std::cout << this->hint_ << ": Depth image columns: " << this->imgCols_ << std::endl;

        // depth image rows
        this->declare_parameter<int>("image_rows", 480);
        this->get_parameter("image_rows", this->imgRows_);
        std::cout << this->hint_ << ": Depth image rows: " << this->imgRows_ << std::endl;
        this->projPoints_.resize(this->imgCols_ * this->imgRows_ / (this->skipPixel_ * this->skipPixel_));
        this->pointsDepth_.resize(this->imgCols_ * this->imgRows_ / (this->skipPixel_ * this->skipPixel_));

        // transform matrix: body to depth camera
        std::vector<double> body2CamVec(16, 0.0);
        this->declare_parameter<std::vector<double>>("body_to_camera_depth", std::vector<double>(16));
        this->get_parameter("body_to_camera_depth", body2CamVec);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                this->body2CamDepth_(i, j) = body2CamVec[i * 4 + j];
            }
        }

        // transform matrix: body to color camera
        std::vector<double> body2CamColorVec(16, 0.0);
        this->declare_parameter<std::vector<double>>("body_to_camera_color", std::vector<double>(16));
        this->get_parameter("body_to_camera_color", body2CamColorVec);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                this->body2CamColor_(i, j) = body2CamColorVec[i * 4 + j];
            }
        }

        // time step
        this->declare_parameter<double>("time_step", 0.033);
        this->get_parameter("time_step", this->dt_);
        std::cout << this->hint_ << ": The time step for the system is set to: " << this->dt_ << std::endl;

        // raycast max length
        this->declare_parameter<double>("raycast_max_length", 5.0);
        this->get_parameter("raycast_max_length", this->raycastMaxLength_);
        std::cout << this->hint_ << ": Raycast max length: " << this->raycastMaxLength_ << std::endl;

        // IOU threshold
        this->declare_parameter<double>("filtering_BBox_IOU_threshold", 0.5);
        this->get_parameter("filtering_BBox_IOU_threshold", this->boxIOUThresh_);
        std::cout << this->hint_ << ": The threshold for bounding box IOU filtering is set to: " << this->boxIOUThresh_ << std::endl;

        // DBSCAN ground height
        this->declare_parameter<double>("ground_height", 0.1);
        this->get_parameter("ground_height", this->groundHeight_);
        std::cout << this->hint_ << ": Ground height is set to: " << this->groundHeight_ << std::endl;

        // DBSCAN minimum number of points in each cluster
        this->declare_parameter<int>("dbscan_min_points_cluster", 18);
        this->get_parameter("dbscan_min_points_cluster", this->dbMinPointsCluster_);
        std::cout << this->hint_ << ": DBSCAN Minimum point in each cluster is set to: " << this->dbMinPointsCluster_ << std::endl;

        // DBSCAN search range
        this->declare_parameter<double>("dbscan_search_range_epsilon", 0.3);
        this->get_parameter("dbscan_search_range_epsilon", this->dbEpsilon_);
        std::cout << this->hint_ << ": DBSCAN epsilon is set to: " << this->dbEpsilon_ << std::endl;

        // DBSCAN min number of points for a voxel to be occupied in voxel filter
        this->declare_parameter<int>("voxel_occupied_thresh", 10);
        this->get_parameter("voxel_occupied_thresh", this->voxelOccThresh_);
        std::cout << this->hint_ << ": Min num of points for a voxel to be occupied in voxel filter is set to: " << this->voxelOccThresh_ << std::endl;

        // tracking history size
        this->declare_parameter<int>("history_size", 5);
        this->get_parameter("history_size", this->histSize_);
        std::cout << this->hint_ << ": The history for tracking is set to: " << this->histSize_ << std::endl;

        // tracking prediction size
        this->declare_parameter<int>("prediction_size", 5);
        this->get_parameter("prediction_size", this->predSize_);
        std::cout << this->hint_ << ": The prediction size is set to: " << this->predSize_ << std::endl;

        // similarity threshold for data association
        this->declare_parameter<double>("similarity_threshold", 0.9);
        this->get_parameter("similarity_threshold", this->simThresh_);
        std::cout << this->hint_ << ": The similarity threshold for data association is set to: " << this->simThresh_ << std::endl;

        // retrack similarity threshold
        this->declare_parameter<double>("retrack_similarity_threshold", 0.5);
        this->get_parameter("retrack_similarity_threshold", this->simThreshRetrack_);
        std::cout << this->hint_ << ": The retrack similarity threshold is set to: " << this->simThreshRetrack_ << std::endl;    
    
        // covariance for Kalman Filter
        this->declare_parameter<double>("e_p", 0.5);
        this->get_parameter("e_p", this->eP_);
        std::cout << this->hint_ << ": The covariance for the Kalman filter is set to: " << this->eP_ << std::endl;

        // noise for prediction in Kalman Filter for position
        this->declare_parameter<double>("e_q_pos", 0.5);
        this->get_parameter("e_q_pos", this->eQPos_);
        std::cout << this->hint_ << ": The noise for prediction for position in the Kalman filter is set to: " << this->eQPos_ << std::endl;

        // noise for prediction in Kalman Filter for velocity
        this->declare_parameter<double>("e_q_vel", 0.5);
        this->get_parameter("e_q_vel", this->eQVel_);
        std::cout << this->hint_ << ": The noise for prediction for velocity in the Kalman filter is set to: " << this->eQVel_ << std::endl;

        // noise for prediction in Kalman Filter for acceleration
        this->declare_parameter<double>("e_q_acc", 0.5);
        this->get_parameter("e_q_acc", this->eQAcc_);
        std::cout << this->hint_ << ": The noise for prediction for acceleration in the Kalman filter is set to: " << this->eQAcc_ << std::endl;

        // noise for measurement for position in Kalman Filter
        this->declare_parameter<double>("e_r_pos", 0.5);
        this->get_parameter("e_r_pos", this->eRPos_);
        std::cout << this->hint_ << ": The noise for measurement for position in Kalman Filter is set to: " << this->eRPos_ << std::endl;

        // noise for measurement for velocity in Kalman Filter
        this->declare_parameter<double>("e_r_vel", 0.5);
        this->get_parameter("e_r_vel", this->eRVel_);
        std::cout << this->hint_ << ": The noise for measurement for velocity in Kalman Filter is set to: " << this->eRVel_ << std::endl;

        // noise for measurement for acceleration in Kalman Filter
        this->declare_parameter<double>("e_r_acc", 0.5);
        this->get_parameter("e_r_acc", this->eRAcc_);
        std::cout << this->hint_ << ": The noise for measurement for acceleration in Kalman Filter is set to: " << this->eRAcc_ << std::endl;
        
        // number of frames used in KF for observation
        this->declare_parameter<int>("kalman_filter_averaging_frames", 10);
        this->get_parameter("kalman_filter_averaging_frames", this->kfAvgFrames_);
        std::cout << this->hint_ << ": Number of frames used in KF for observation is set to: " << this->kfAvgFrames_ << std::endl;

        // frame skip
        this->declare_parameter<int>("frame_skip", 5);
        this->get_parameter("frame_skip", this->skipFrame_);
        std::cout << this->hint_ << ": The number of frames skipped for classification is set to: " << this->skipFrame_ << std::endl;
    
        // dynamic velocity threshold
        this->declare_parameter<double>("dynamic_velocity_threshold", 0.35);
        this->get_parameter("dynamic_velocity_threshold", this->dynaVelThresh_);
        std::cout << this->hint_ << ": The velocity threshold for dynamic classification is set to: " << this->dynaVelThresh_ << std::endl;

        // dynamic voting threshold
        this->declare_parameter<double>("dynamic_voting_threshold", 0.8);
        this->get_parameter("dynamic_voting_threshold", this->dynaVoteThresh_);
        std::cout << this->hint_ << ": The voting threshold for dynamic classification is set to: " << this->dynaVoteThresh_ << std::endl;

        // maximum skip ratio
        this->declare_parameter<double>("maximum_skip_ratio", 0.5);
        this->get_parameter("maximum_skip_ratio", this->maxSkipRatio_);
        std::cout << this->hint_ << ": The upper limit of points skipping in classification is set to: " << this->maxSkipRatio_ << std::endl;

        // history threshold for fixing box size
        this->declare_parameter<int>("fix_size_history_threshold", 10);
        this->get_parameter("fix_size_history_threshold", this->fixSizeHistThresh_);
        std::cout << this->hint_ << ": History threshold for fixing size is set to: " << this->fixSizeHistThresh_ << std::endl;

        // dimension threshold for fixing box size
        this->declare_parameter<double>("fix_size_dimension_threshold", 0.4);
        this->get_parameter("fix_size_dimension_threshold", this->fixSizeDimThresh_);
        std::cout << this->hint_ << ": Dimension threshold for fixing size is set to: " << this->fixSizeDimThresh_ << std::endl;
        
        // frames to force dynamic
        this->declare_parameter<int>("frames_force_dynamic", 20);
        this->get_parameter("frames_force_dynamic", this->forceDynaFrames_);
        std::cout << this->hint_ << ": Range of searching dynamic obstacles in box history is set to: " << this->forceDynaFrames_ << std::endl;

        // threshold for forcing dynamic obstacles
        this->declare_parameter<int>("frames_force_dynamic_check_range", 30);
        this->get_parameter("frames_force_dynamic_check_range", this->forceDynaCheckRange_);
        std::cout << this->hint_ << ": Threshold for forcing dynamic obstacles is set to: " << this->forceDynaCheckRange_ << std::endl;

        // dynamic consistency check
        this->declare_parameter<int>("dynamic_consistency_threshold", 3);
        this->get_parameter("dynamic_consistency_threshold", this->dynamicConsistThresh_);
        std::cout << this->hint_ << ": Threshold for dynamic consistency check is set to: " << this->dynamicConsistThresh_ << std::endl;

        // constrain target object size
        this->declare_parameter<bool>("constrain_size", false);
        this->get_parameter("constrain_size", this->constrainSize_);
        std::cout << this->hint_ << ": Target object constrain is set to: " << this->constrainSize_ << std::endl;

        // object target sizes
        std::vector<double> targetObjectSizeTemp;
        this->declare_parameter<std::vector<double>>("target_object_size", std::vector<double>());
        this->get_parameter("target_object_size", targetObjectSizeTemp);
        if (targetObjectSizeTemp.empty()) {
            std::cout << this->hint_ << ": No target object size found. Do not apply target object size." << std::endl;
        } else {
            for (size_t i = 0; i < targetObjectSizeTemp.size(); i += 3) {
                Eigen::Vector3d targetSize(targetObjectSizeTemp[i], targetObjectSizeTemp[i + 1], targetObjectSizeTemp[i + 2]);
                this->targetObjectSize_.push_back(targetSize);
                std::cout << this->hint_ << ": Target object size is set to: [" << targetObjectSizeTemp[i] << ", " << targetObjectSizeTemp[i + 1] << ", " << targetObjectSizeTemp[i + 2] << "]." << std::endl;
            }
        }
    }

    void dynamicDetector::registerPub(){
        // uv detector depth map pub
        this->uvDepthMapPub_ = this->create_publisher<sensor_msgs::msg::Image>(this->ns_ + "/detected_depth_map", 10);

        // // uv detector u depth map pub
        this->uDepthMapPub_ = this->create_publisher<sensor_msgs::msg::Image>(this->ns_ + "/detected_u_depth_map", 10);

        // // uv detector bird view pub
        this->uvBirdViewPub_ = this->create_publisher<sensor_msgs::msg::Image>(this->ns_ + "/bird_eye_view", 10);

        // // color 2D bounding boxes pub
        this->detectedColorImgPub_ = this->create_publisher<sensor_msgs::msg::Image>(this->ns_ + "/detected_color_image", 10);

        // uv detector bounding box pub
        this->uvBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/uv_bboxes", 10);

        // DBSCAN bounding box pub
        this->dbBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/dbscan_bboxes", 10);

        // filtered bounding box pub
        this->filteredBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/filtered_bboxes", 10);

        // tracked bounding box pub
        this->trackedBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/tracked_bboxes", 10);

        // history trajectory pub
        this->dynamicBBoxesPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/dynamic_bboxes", 10);

        // dynamic bounding box pub
        this->historyTrajPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/history_trajectories", 10);

        // velocity visualization pub
        this->velVisPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/velocity_visualizaton", 10);

        // filtered pointcloud pub
        this->filteredPointsPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->ns_ + "/filtered_depth_cloud", 10);

        // dynamic pointcloud pub
        this->dynamicPointsPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->ns_ + "/dynamic_point_cloud", 10);
    }


    void dynamicDetector::registerCallback(){
		this->depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->depthTopicName_);
        if (this->localizationMode_ == 0){
			this->poseSub_ = std::make_shared<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>>(this, this->poseTopicName_);
			this->depthPoseSync_  = std::make_shared<message_filters::Synchronizer<depthPoseSync>>(depthPoseSync(100), *this->depthSub_, *this->poseSub_);
			this->depthPoseSync_->registerCallback(std::bind(&dynamicDetector::depthPoseCB, this, std::placeholders::_1, std::placeholders::_2));
        }
        else{
			this->odomSub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, this->odomTopicName_);
			this->depthOdomSync_ = std::make_shared<message_filters::Synchronizer<depthOdomSync>>(depthOdomSync(100), *this->depthSub_, *this->odomSub_);
			this->depthOdomSync_->registerCallback(std::bind(&dynamicDetector::depthOdomCB, this, std::placeholders::_1, std::placeholders::_2));
        }

        // color image subscriber
        this->colorImageSub_ = this->create_subscription<sensor_msgs::msg::Image>(this->colorImgTopicName_, 10, std::bind(&dynamicDetector::colorImgCB, this, std::placeholders::_1));

        // yolo detection results subscriber
        this->yoloDetectionSub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>("yolo_detector/detected_bounding_boxes", 10, std::bind(&dynamicDetector::yoloDetectionCB, this, std::placeholders::_1));
    
        int timeStep = this->dt_ * 1000.0;
        // detection timer
        this->detectionTimer_ = this->create_wall_timer(std::chrono::milliseconds(timeStep), std::bind(&dynamicDetector::detectionCB, this));

        // tracking timer
        this->trackingTimer_ = this->create_wall_timer(std::chrono::milliseconds(timeStep), std::bind(&dynamicDetector::trackingCB, this));

        // classification timer
        this->classificationTimer_ = this->create_wall_timer(std::chrono::milliseconds(timeStep), std::bind(&dynamicDetector::classificationCB, this));

        // visualization timer
        this->visTimer_ = this->create_wall_timer(33ms, std::bind(&dynamicDetector::visCB, this));

		// get dynamic obstacle service
        this->getDynamicObstacleServer_ = this->create_service<onboard_detector::srv::GetDynamicObstacles>(this->ns_ + "/get_dynamic_obstacles", std::bind(&dynamicDetector::getDynamicObstaclesSrv, this, std::placeholders::_1, std::placeholders::_2));	
    }

    bool dynamicDetector::getDynamicObstaclesSrv(const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Request> req, 
                                                 const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Response> res){
        // Get the current robot position
        Eigen::Vector3d currPos = Eigen::Vector3d (req->current_position.x, req->current_position.y, req->current_position.z);

        // Vector to store obstacles along with their distances
        std::vector<std::pair<double, onboardDetector::box3D>> obstaclesWithDistances;

        // Go through all obstacles and calculate distances
        for (const onboardDetector::box3D& bbox : this->dynamicBBoxes_) {
            Eigen::Vector3d obsPos(bbox.x, bbox.y, bbox.z);
            Eigen::Vector3d diff = currPos - obsPos;
            diff(2) = 0.;
            double distance = diff.norm();
            if (distance <= req->range) {
                obstaclesWithDistances.push_back(std::make_pair(distance, bbox));
            }
        }

        // Sort obstacles by distance in ascending order
        std::sort(obstaclesWithDistances.begin(), obstaclesWithDistances.end(), 
                [](const std::pair<double, onboardDetector::box3D>& a, const std::pair<double, onboardDetector::box3D>& b) {
                    return a.first < b.first;
                });

        // Push sorted obstacles into the response
        for (const auto& item : obstaclesWithDistances) {
            const onboardDetector::box3D& bbox = item.second;

            geometry_msgs::msg::Vector3 pos;
            geometry_msgs::msg::Vector3 vel;
            geometry_msgs::msg::Vector3 size;

            pos.x = bbox.x;
            pos.y = bbox.y;
            pos.z = bbox.z;

            vel.x = bbox.Vx;
            vel.y = bbox.Vy;
            vel.z = 0.;

            size.x = bbox.x_width;
            size.y = bbox.y_width;
            size.z = bbox.z_width;

            res->position.push_back(pos);
            res->velocity.push_back(vel);
            res->size.push_back(size);
        }
        return true;
    }

    void dynamicDetector::depthPoseCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose){
        // store current depth image
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1){
            (imgPtr->image).convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
        }
        imgPtr->image.copyTo(this->depthImage_);

		// store current robot position and orientation
		this->position_(0) = pose->pose.position.x;
		this->position_(1) = pose->pose.position.y;
		this->position_(2) = pose->pose.position.z;
      	Eigen::Quaterniond robotQuat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
		this->orientation_ = robotQuat.toRotationMatrix();

        // store current position and orientation (camera)
        Eigen::Matrix4d camPoseDepthMatrix, camPoseColorMatrix;
        this->getSensorPose(pose, this->body2CamDepth_, camPoseDepthMatrix);
        this->getSensorPose(pose, this->body2CamColor_, camPoseColorMatrix);

        this->positionDepth_(0) = camPoseDepthMatrix(0, 3);
        this->positionDepth_(1) = camPoseDepthMatrix(1, 3);
        this->positionDepth_(2) = camPoseDepthMatrix(2, 3);
        this->orientationDepth_ = camPoseDepthMatrix.block<3, 3>(0, 0);

        this->positionColor_(0) = camPoseColorMatrix(0, 3);
        this->positionColor_(1) = camPoseColorMatrix(1, 3);
        this->positionColor_(2) = camPoseColorMatrix(2, 3);
        this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::depthOdomCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const nav_msgs::msg::Odometry::ConstSharedPtr& odom){
        // store current depth image
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1){
            (imgPtr->image).convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
        }
        imgPtr->image.copyTo(this->depthImage_);

		// store current robot position and orientation
		this->position_(0) = odom->pose.pose.position.x;
		this->position_(1) = odom->pose.pose.position.y;
		this->position_(2) = odom->pose.pose.position.z;
      	Eigen::Quaterniond robotQuat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
		this->orientation_ = robotQuat.toRotationMatrix();

        // store current position and orientation (camera)
        Eigen::Matrix4d camPoseDepthMatrix, camPoseColorMatrix;
        this->getSensorPose(odom, this->body2CamDepth_, camPoseDepthMatrix);
        this->getSensorPose(odom, this->body2CamColor_, camPoseColorMatrix);

        this->positionDepth_(0) = camPoseDepthMatrix(0, 3);
        this->positionDepth_(1) = camPoseDepthMatrix(1, 3);
        this->positionDepth_(2) = camPoseDepthMatrix(2, 3);
        this->orientationDepth_ = camPoseDepthMatrix.block<3, 3>(0, 0);

        this->positionColor_(0) = camPoseColorMatrix(0, 3);
        this->positionColor_(1) = camPoseColorMatrix(1, 3);
        this->positionColor_(2) = camPoseColorMatrix(2, 3);
        this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::colorImgCB(const sensor_msgs::msg::Image::ConstSharedPtr& img){
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        imgPtr->image.copyTo(this->detectedColorImage_);
    }

    void dynamicDetector::yoloDetectionCB(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections){
        this->yoloDetectionResults_ = *detections;
    }

    void dynamicDetector::detectionCB(){
        // update pose history
        this->updatePoseHist();
        this->dbscanDetect();
        this->uvDetect();
        this->filterBBoxes();
        this->newDetectFlag_ = true; // get a new detection
    }

    void dynamicDetector::trackingCB(){
        std::vector<int> bestMatch; // for each current detection, which index of previous obstacle match
        std::vector<int> boxOOR; // whether the box in hist is detected in this time step
        // bestMatch: an array of current detected object num size and the entries are idx from the history
        // boxOOR: an array of hist bbox size, entries: 0 or 1 (unmatched)
        this->boxAssociation(bestMatch, boxOOR);
        // kalman filter tracking
        if (bestMatch.size() or boxOOR.size()){
            this->kalmanFilterAndUpdateHist(bestMatch, boxOOR);
        }
        else {
            this->boxHist_.clear();
            this->pcHist_.clear();
        }
    }

    void dynamicDetector::classificationCB(){
        std::vector<onboardDetector::box3D> dynamicBBoxesTemp;

        // Iterate through all pointcloud/bounding boxes history (note that yolo's pointclouds are dummy pointcloud (empty))
        // NOTE: There are 3 cases which we don't need to perform dynamic obstacle identification.
        // cout << "======================" << endl;
        for (size_t i=0; i<this->pcHist_.size() ; ++i){
            // ===================================================================================
            // CASE 0: predicted dynamic obstacle
            if (this->boxHist_[i][0].is_estimated){
                onboardDetector::box3D estimatedBBox;
                this->getEstimateBox(this->boxHist_[i], estimatedBBox);
                if (this->constrainSize_){
                    bool findMatch = false;
                    for (Eigen::Vector3d targetSize : this->targetObjectSize_){
                        double xdiff = std::abs(this->boxHist_[i][0].x_width - targetSize(0));
                        double ydiff = std::abs(this->boxHist_[i][0].y_width - targetSize(1));
                        double zdiff = std::abs(this->boxHist_[i][0].z_width - targetSize(2)); 
                        if (xdiff < 0.8 and ydiff < 0.8 and zdiff < 1.0){
                            findMatch = true;
                        }
                    }
                    if (findMatch){
                        dynamicBBoxesTemp.push_back(estimatedBBox);
                    }
                }
                else{
                    dynamicBBoxesTemp.push_back(estimatedBBox);
                }
                
                continue; 
            }

            // ===================================================================================
            // CASE I: yolo recognized as dynamic dynamic obstacle
            // cout << "box: " << i <<  "x y z: " << this->boxHist_[i][0].x << " " << this->boxHist_[i][0].y << " " << this->boxHist_[i][0].z << endl;
            // cout << "box is human: " << this->boxHist_[i][0].is_human << endl;
            if (this->boxHist_[i][0].is_human){
                dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
                continue;
            }
            // ===================================================================================


            // ===================================================================================
            // CASE II: history length is not enough to run classification
            int curFrameGap;
            if (int(this->pcHist_[i].size()) < this->skipFrame_+1){
                curFrameGap = this->pcHist_[i].size() - 1;
            }
            else{
                curFrameGap = this->skipFrame_;
            }
            // ===================================================================================


            // ==================================================================================
            // CASE III: Force Dynamic (if the obstacle is classifed as dynamic for several time steps)
            int dynaFrames = 0;
            if (int(this->boxHist_[i].size()) > this->forceDynaCheckRange_){
                for (int j=1 ; j<this->forceDynaCheckRange_+1 ; ++j){
                    if (this->boxHist_[i][j].is_dynamic){
                        ++dynaFrames;
                    }
                }
            }

            if (dynaFrames >= this->forceDynaFrames_){
                this->boxHist_[i][0].is_dynamic = true;
                dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
                continue;
            }
            // ===================================================================================

            std::vector<Eigen::Vector3d> currPc = this->pcHist_[i][0];
            std::vector<Eigen::Vector3d> prevPc = this->pcHist_[i][curFrameGap];
            Eigen::Vector3d Vcur(0.,0.,0.); // single point velocity 
            Eigen::Vector3d Vbox(0.,0.,0.); // bounding box velocity 
            Eigen::Vector3d Vkf(0.,0.,0.);  // velocity estimated from kalman filter
            int numPoints = currPc.size(); // it changes within loop
            int votes = 0;

            Vbox(0) = (this->boxHist_[i][0].x - this->boxHist_[i][curFrameGap].x)/(this->dt_*curFrameGap);
            Vbox(1) = (this->boxHist_[i][0].y - this->boxHist_[i][curFrameGap].y)/(this->dt_*curFrameGap);
            Vbox(2) = (this->boxHist_[i][0].z - this->boxHist_[i][curFrameGap].z)/(this->dt_*curFrameGap);
            Vkf(0) = this->boxHist_[i][0].Vx;
            Vkf(1) = this->boxHist_[i][0].Vy;

            // find nearest neighbor
            int numSkip = 0;
            for (size_t j=0 ; j<currPc.size() ; ++j){
                // don't perform classification for points unseen in previous frame
                if (!this->isInFov(this->positionHist_[curFrameGap], this->orientationHist_[curFrameGap], currPc[j])){
                    ++numSkip;
                    --numPoints;
                    continue;
                }

                double minDist = 2;
                Eigen::Vector3d nearestVect;
                for (size_t k=0 ; k<prevPc.size() ; k++){ // find the nearest point in the previous pointcloud
                    double dist = (currPc[j]-prevPc[k]).norm();
                    if (abs(dist) < minDist){
                        minDist = dist;
                        nearestVect = currPc[j]-prevPc[k];
                    }
                }
                Vcur = nearestVect/(this->dt_*curFrameGap); Vcur(2) = 0;
                double velSim = Vcur.dot(Vbox)/(Vcur.norm()*Vbox.norm());

                if (velSim < 0){
                    ++numSkip;
                    --numPoints;
                }
                else{
                    if (Vcur.norm()>this->dynaVelThresh_){
                        ++votes;
                    }
                }
            }
            
            
            // update dynamic boxes
            double voteRatio = (numPoints>0)?double(votes)/double(numPoints):0;
            double velNorm = Vkf.norm();

            // voting and velocity threshold
            // 1. point cloud voting ratio.
            // 2. velocity (from kalman filter) 
            // 3. enough valid point correspondence 
            if (voteRatio>=this->dynaVoteThresh_ && velNorm>=this->dynaVelThresh_ && double(numSkip)/double(numPoints)<this->maxSkipRatio_){
                this->boxHist_[i][0].is_dynamic_candidate = true;
                // dynamic-consistency check
                int dynaConsistCount = 0;
                if (int(this->boxHist_[i].size()) >= this->dynamicConsistThresh_){
                    for (int j=0 ; j<this->dynamicConsistThresh_; ++j){
                        if (this->boxHist_[i][j].is_dynamic_candidate){
                            ++dynaConsistCount;
                        }
                    }
                }            
                if (dynaConsistCount == this->dynamicConsistThresh_){
                    // set as dynamic and push into history
                    this->boxHist_[i][0].is_dynamic = true;
                    dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);    
                }
            }
        }

        // filter the dynamic obstacles based on the target sizes
        if (this->constrainSize_){
            std::vector<onboardDetector::box3D> dynamicBBoxesBeforeConstrain = dynamicBBoxesTemp;
            dynamicBBoxesTemp.clear();

            for (onboardDetector::box3D ob : dynamicBBoxesBeforeConstrain){
                if (not ob.is_estimated){
                    bool findMatch = false;
                    for (Eigen::Vector3d targetSize : this->targetObjectSize_){
                        double xdiff = std::abs(ob.x_width - targetSize(0));
                        double ydiff = std::abs(ob.y_width - targetSize(1));
                        double zdiff = std::abs(ob.z_width - targetSize(2)); 
                        if (xdiff < 0.8 and ydiff < 0.8 and zdiff < 1.0){
                            findMatch = true;
                        }
                    }

                    if (findMatch){
                        dynamicBBoxesTemp.push_back(ob);
                    }
                }
                else{
                    dynamicBBoxesTemp.push_back(ob);
                }
            }
        }

        this->dynamicBBoxes_ = dynamicBBoxesTemp;
    }

    void dynamicDetector::visCB(){
        this->publishUVImages();
        this->publish3dBox(this->uvBBoxes_, this->uvBBoxesPub_, 0, 1, 0);
        std::vector<Eigen::Vector3d> dynamicPoints;
        this->getDynamicPc(dynamicPoints);
        this->publishPoints(dynamicPoints, this->dynamicPointsPub_);
        this->publishPoints(this->filteredPoints_, this->filteredPointsPub_);
        this->publish3dBox(this->dbBBoxes_, this->dbBBoxesPub_, 1, 0, 0);
        this->publishColorImages();
        this->publish3dBox(this->filteredBBoxes_, this->filteredBBoxesPub_, 0, 1, 1);
        this->publish3dBox(this->trackedBBoxes_, this->trackedBBoxesPub_, 1, 1, 0);
        this->publish3dBox(this->dynamicBBoxes_, this->dynamicBBoxesPub_, 0, 0, 1);
        this->publishHistoryTraj();
        this->publishVelVis();
    }

    void dynamicDetector::dbscanDetect(){
        // 1. get pointcloud
        this->projectDepthImage();

        // 2. filter points
        this->filterPoints(this->projPoints_, this->filteredPoints_);

        // 3. cluster points and get bounding boxes
        this->clusterPointsAndBBoxes(this->filteredPoints_, this->dbBBoxes_, this->pcClusters_, this->pcClusterCenters_, this->pcClusterStds_);
    }

    void dynamicDetector::uvDetect(){
        // initialization
        if (this->uvDetector_ == NULL){
            this->uvDetector_.reset(new UVdetector ());
            this->uvDetector_->fx = this->fx_;
            this->uvDetector_->fy = this->fy_;
            this->uvDetector_->px = this->cx_;
            this->uvDetector_->py = this->cy_;
            this->uvDetector_->depthScale_ = this->depthScale_; 
            this->uvDetector_->max_dist = this->raycastMaxLength_ * 1000;
        }

        // detect from depth mapcalBox
        if (not this->depthImage_.empty()){
            this->uvDetector_->depth = this->depthImage_;
            this->uvDetector_->detect();
            this->uvDetector_->extract_3Dbox();

            this->uvDetector_->display_U_map();
            this->uvDetector_->display_bird_view();
            this->uvDetector_->display_depth();

            // transform to the world frame (recalculate the boudning boxes)
            std::vector<onboardDetector::box3D> uvBBoxes;
            this->transformUVBBoxes(uvBBoxes);
            this->uvBBoxes_ = uvBBoxes;
        }
    }

    void dynamicDetector::filterBBoxes(){
        std::vector<onboardDetector::box3D> filteredBBoxesTemp;
        std::vector<std::vector<Eigen::Vector3d>> filteredPcClustersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterCentersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterStdsTemp; 
        // find best IOU match for both uv and dbscan. If they are best for each other, then add to filtered bbox and fuse.
        for (size_t i=0 ; i<this->uvBBoxes_.size(); ++i){
            onboardDetector::box3D uvBBox = this->uvBBoxes_[i];
            double bestIOUForUVBBox, bestIOUForDBBBox;
            int bestMatchForUVBBox = this->getBestOverlapBBox(uvBBox, this->dbBBoxes_, bestIOUForUVBBox);
            if (bestMatchForUVBBox == -1) continue; // no match at all
            onboardDetector::box3D matchedDBBBox = this->dbBBoxes_[bestMatchForUVBBox]; 
            std::vector<Eigen::Vector3d> matchedPcCluster = this->pcClusters_[bestMatchForUVBBox];
            Eigen::Vector3d matchedPcClusterCenter = this->pcClusterCenters_[bestMatchForUVBBox];
            Eigen::Vector3d matchedPcClusterStd = this->pcClusterStds_[bestMatchForUVBBox];
            int bestMatchForDBBBox = this->getBestOverlapBBox(matchedDBBBox, this->uvBBoxes_, bestIOUForDBBBox);

            // if best match is each other and both the IOU is greater than the threshold
            if (bestMatchForDBBBox == int(i) and bestIOUForUVBBox > this->boxIOUThresh_ and bestIOUForDBBBox > this->boxIOUThresh_){
                onboardDetector::box3D bbox;
                
                // take concervative strategy
                double xmax = std::max(uvBBox.x+uvBBox.x_width/2, matchedDBBBox.x+matchedDBBBox.x_width/2);
                double xmin = std::min(uvBBox.x-uvBBox.x_width/2, matchedDBBBox.x-matchedDBBBox.x_width/2);
                double ymax = std::max(uvBBox.y+uvBBox.y_width/2, matchedDBBBox.y+matchedDBBBox.y_width/2);
                double ymin = std::min(uvBBox.y-uvBBox.y_width/2, matchedDBBBox.y-matchedDBBBox.y_width/2);
                double zmax = std::max(uvBBox.z+uvBBox.z_width/2, matchedDBBBox.z+matchedDBBBox.z_width/2);
                double zmin = std::min(uvBBox.z-uvBBox.z_width/2, matchedDBBBox.z-matchedDBBBox.z_width/2);
                bbox.x = (xmin+xmax)/2;
                bbox.y = (ymin+ymax)/2;
                bbox.z = (zmin+zmax)/2;
                bbox.x_width = xmax-xmin;
                bbox.y_width = ymax-ymin;
                bbox.z_width = zmax-zmin;
                bbox.Vx = 0;
                bbox.Vy = 0;

                filteredBBoxesTemp.push_back(bbox);
                filteredPcClustersTemp.push_back(matchedPcCluster);      
                filteredPcClusterCentersTemp.push_back(matchedPcClusterCenter);
                filteredPcClusterStdsTemp.push_back(matchedPcClusterStd);
            }
        }


        // Instead of using YOLO for ensembling, only use YOLO for dynamic object identification
        // For each 2D YOLO detected bounding box, find the best match projected 2D bounding boxes
        if (this->yoloDetectionResults_.detections.size() != 0){
            // Project 2D bbox in color image plane from 3D
            vision_msgs::msg::Detection2DArray filteredDetectionResults;
            for (int j=0; j<int(filteredBBoxesTemp.size()); ++j){
                onboardDetector::box3D bbox = filteredBBoxesTemp[j];

                // 1. transform the bounding boxes into the camera frame
                Eigen::Vector3d centerWorld (bbox.x, bbox.y, bbox.z);
                Eigen::Vector3d sizeWorld (bbox.x_width, bbox.y_width, bbox.z_width);
                Eigen::Vector3d centerCam, sizeCam;
                this->transformBBox(centerWorld, sizeWorld, -this->orientationColor_.inverse() * this->positionColor_, this->orientationColor_.inverse(), centerCam, sizeCam);


                // 2. find the top left and bottom right corner 3D position of the transformed bbox
                Eigen::Vector3d topleft (centerCam(0)-sizeCam(0)/2, centerCam(1)-sizeCam(1)/2, centerCam(2));
                Eigen::Vector3d bottomright (centerCam(0)+sizeCam(0)/2, centerCam(1)+sizeCam(1)/2, centerCam(2));

                // 3. project those two points into the camera image plane
                int tlX = (this->fxC_ * topleft(0) + this->cxC_ * topleft(2)) / topleft(2);
                int tlY = (this->fyC_ * topleft(1) + this->cyC_ * topleft(2)) / topleft(2);
                int brX = (this->fxC_ * bottomright(0) + this->cxC_ * bottomright(2)) / bottomright(2);
                int brY = (this->fyC_ * bottomright(1) + this->cyC_ * bottomright(2)) / bottomright(2);

                vision_msgs::msg::Detection2D result;
                result.bbox.center.position.x = tlX;
                result.bbox.center.position.y = tlY;
                result.bbox.size_x = brX - tlX;
                result.bbox.size_y = brY - tlY;
                filteredDetectionResults.detections.push_back(result);

                cv::Rect bboxVis;
                bboxVis.x = tlX;
                bboxVis.y = tlY;
                bboxVis.height = brY - tlY;
                bboxVis.width = brX - tlX;
                cv::rectangle(this->detectedColorImage_, bboxVis, cv::Scalar(0, 255, 0), 5, 8, 0);
            }


            for (int i=0; i<int(this->yoloDetectionResults_.detections.size()); ++i){
                int tlXTarget = int(this->yoloDetectionResults_.detections[i].bbox.center.position.x);
                int tlYTarget = int(this->yoloDetectionResults_.detections[i].bbox.center.position.y);
                int brXTarget = tlXTarget + int(this->yoloDetectionResults_.detections[i].bbox.size_x);
                int brYTarget = tlYTarget + int(this->yoloDetectionResults_.detections[i].bbox.size_y);

                cv::Rect bboxVis;
                bboxVis.x = tlXTarget;
                bboxVis.y = tlYTarget;
                bboxVis.height = brYTarget - tlYTarget;
                bboxVis.width = brXTarget - tlXTarget;
                cv::rectangle(this->detectedColorImage_, bboxVis, cv::Scalar(255, 0, 0), 5, 8, 0);

                // Define the text to be added
                std::string text = "dynamic";

                // Define the position for the text (above the bounding box)
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 1.0;
                int thickness = 2;
                int baseline;
                cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
                cv::Point textOrg(bboxVis.x, bboxVis.y - 10);  // 10 pixels above the bounding box

                // Add the text to the image
                cv::putText(this->detectedColorImage_, text, textOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, 8);

                double bestIOU = 0.0;
                int bestIdx = -1;
                for (int j=0; j<int(filteredBBoxesTemp.size()); ++j){
                    int tlX = int(filteredDetectionResults.detections[j].bbox.center.position.x);
                    int tlY = int(filteredDetectionResults.detections[j].bbox.center.position.y);
                    int brX = tlX + int(filteredDetectionResults.detections[j].bbox.size_x);
                    int brY = tlY + int(filteredDetectionResults.detections[j].bbox.size_y);
                    
                    // check the IOU between yolo and projected bbox
                    double xOverlap = double(std::max(0, std::min(brX, brXTarget) - std::max(tlX, tlXTarget)));
                    double yOverlap = double(std::max(0, std::min(brY, brYTarget) - std::max(tlY, tlYTarget)));
                    double intersection = xOverlap * yOverlap;

                    // Calculate union area
                    double areaBox = double((brX - tlX) * (brY - tlY));
                    double areaBoxTarget = double((brXTarget - tlXTarget) * (brYTarget - tlYTarget));
                    double unionArea = areaBox + areaBoxTarget - intersection;
                    double IOU = (unionArea == 0) ? 0 : intersection / unionArea;

                    if (IOU > bestIOU){
                        bestIOU = IOU;
                        bestIdx = j;
                    }
                }

                if (bestIOU > 0.5){
                    filteredBBoxesTemp[bestIdx].is_dynamic = true;
                    filteredBBoxesTemp[bestIdx].is_human = true;
                }
            }
        }

        this->filteredBBoxes_ = filteredBBoxesTemp;
        this->filteredPcClusters_ = filteredPcClustersTemp;
        this->filteredPcClusterCenters_ = filteredPcClusterCentersTemp;
        this->filteredPcClusterStds_ = filteredPcClusterStdsTemp;
    }

    void dynamicDetector::projectDepthImage(){
        this->projPointsNum_ = 0;

        int cols = this->depthImage_.cols;
        int rows = this->depthImage_.rows;
        uint16_t* rowPtr;

        Eigen::Vector3d currPointCam, currPointMap;
        double depth;
        const double inv_factor = 1.0 / this->depthScale_;
        const double inv_fx = 1.0 / this->fx_;
        const double inv_fy = 1.0 / this->fy_;

        // iterate through each pixel in the depth image
        for (int v=this->depthFilterMargin_; v<rows-this->depthFilterMargin_; v=v+this->skipPixel_){ // row
            rowPtr = this->depthImage_.ptr<uint16_t>(v) + this->depthFilterMargin_;
            for (int u=this->depthFilterMargin_; u<cols-this->depthFilterMargin_; u=u+this->skipPixel_){ // column
                depth = (*rowPtr) * inv_factor;
                
                if (*rowPtr == 0) {
                    depth = this->raycastMaxLength_ + 0.1;
                } else if (depth < this->depthMinValue_) {
                    continue;
                } else if (depth > this->depthMaxValue_) {
                    depth = this->raycastMaxLength_ + 0.1;
                }
                rowPtr =  rowPtr + this->skipPixel_;

                // get 3D point in camera frame
                currPointCam(0) = (u - this->cx_) * depth * inv_fx;
                currPointCam(1) = (v - this->cy_) * depth * inv_fy;
                currPointCam(2) = depth;
                currPointMap = this->orientationDepth_ * currPointCam + this->positionDepth_; // transform to map coordinate

                this->projPoints_[this->projPointsNum_] = currPointMap;
                this->pointsDepth_[this->projPointsNum_] = depth;
                this->projPointsNum_ = this->projPointsNum_ + 1;
            }
        } 
    }

    void dynamicDetector::filterPoints(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
        // currently there is only one filtered (might include more in the future)
        std::vector<Eigen::Vector3d> voxelFilteredPoints;
        this->voxelFilter(points, voxelFilteredPoints);
        filteredPoints = voxelFilteredPoints;
    }

    void dynamicDetector::clusterPointsAndBBoxes(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::box3D>& bboxes, std::vector<std::vector<Eigen::Vector3d>>& pcClusters, std::vector<Eigen::Vector3d>& pcClusterCenters, std::vector<Eigen::Vector3d>& pcClusterStds){
        std::vector<onboardDetector::Point> pointsDB;
        this->eigenToDBPointVec(points, pointsDB, points.size());

        this->dbCluster_.reset(new DBSCAN (this->dbMinPointsCluster_, this->dbEpsilon_, pointsDB));

        // DBSCAN clustering
        this->dbCluster_->run();

        // get the cluster data with bounding boxes
        // iterate through all the clustered points and find number of clusters
        int clusterNum = 0;
        for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
            onboardDetector::Point pDB = this->dbCluster_->m_points[i];
            if (pDB.clusterID > clusterNum){
                clusterNum = pDB.clusterID;
            }
        }

        pcClusters.clear();
        pcClusters.resize(clusterNum);
        for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
            onboardDetector::Point pDB = this->dbCluster_->m_points[i];
            if (pDB.clusterID > 0){
                Eigen::Vector3d p = this->dbPointToEigen(pDB);
                pcClusters[pDB.clusterID-1].push_back(p);
            }            
        }

        for (size_t i=0 ; i<pcClusters.size() ; ++i){
            Eigen::Vector3d pcClusterCenter(0.,0.,0.);
            Eigen::Vector3d pcClusterStd(0.,0.,0.);
            this->calcPcFeat(pcClusters[i], pcClusterCenter, pcClusterStd);
            pcClusterCenters.push_back(pcClusterCenter);
            pcClusterStds.push_back(pcClusterStd);
        }

        // calculate the bounding boxes based on the clusters
        bboxes.clear();
        // bboxes.resize(clusterNum);
        for (size_t i=0; i<pcClusters.size(); ++i){
            onboardDetector::box3D box;

            double xmin = pcClusters[i][0](0);
            double ymin = pcClusters[i][0](1);
            double zmin = pcClusters[i][0](2);
            double xmax = pcClusters[i][0](0);
            double ymax = pcClusters[i][0](1);
            double zmax = pcClusters[i][0](2);
            for (size_t j=0; j<pcClusters[i].size(); ++j){
                xmin = (pcClusters[i][j](0)<xmin)?pcClusters[i][j](0):xmin;
                ymin = (pcClusters[i][j](1)<ymin)?pcClusters[i][j](1):ymin;
                zmin = (pcClusters[i][j](2)<zmin)?pcClusters[i][j](2):zmin;
                xmax = (pcClusters[i][j](0)>xmax)?pcClusters[i][j](0):xmax;
                ymax = (pcClusters[i][j](1)>ymax)?pcClusters[i][j](1):ymax;
                zmax = (pcClusters[i][j](2)>zmax)?pcClusters[i][j](2):zmax;
            }
            box.id = i;

            box.x = (xmax + xmin)/2.0;
            box.y = (ymax + ymin)/2.0;
            box.z = (zmax + zmin)/2.0;
            box.x_width = (xmax - xmin)>0.1?(xmax-xmin):0.1;
            box.y_width = (ymax - ymin)>0.1?(ymax-ymin):0.1;
            box.z_width = (zmax - zmin);
            bboxes.push_back(box);
        }
    }

    void dynamicDetector::voxelFilter(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
        const double res = 0.1; // resolution of voxel
        int xVoxels = ceil(2*this->localSensorRange_(0)/res); int yVoxels = ceil(2*this->localSensorRange_(1)/res); int zVoxels = ceil(2*this->localSensorRange_(2)/res);
        int totalVoxels = xVoxels * yVoxels * zVoxels;
        // std::vector<bool> voxelOccupancyVec (totalVoxels, false);
        std::vector<int> voxelOccupancyVec (totalVoxels, 0);

        // Iterate through each points in the cloud
        filteredPoints.clear();
        
        for (int i=0; i<this->projPointsNum_; ++i){
            Eigen::Vector3d p = points[i];

            if (this->isInFilterRange(p) and p(2) >= this->groundHeight_ and this->pointsDepth_[i] <= this->raycastMaxLength_){
                // find the corresponding voxel id in the vector and check whether it is occupied
                int pID = this->posToAddress(p, res);

                // add one point
                voxelOccupancyVec[pID] +=1;

                // add only if thresh points are found
                if (voxelOccupancyVec[pID] == this->voxelOccThresh_){
                    filteredPoints.push_back(p);
                }
            }
        }  
    }

    void dynamicDetector::calcPcFeat(const std::vector<Eigen::Vector3d>& pcCluster, Eigen::Vector3d& pcClusterCenter, Eigen::Vector3d& pcClusterStd){
        int numPoints = pcCluster.size();
        
        // center
        for (int i=0 ; i<numPoints ; i++){
            pcClusterCenter(0) += pcCluster[i](0)/numPoints;
            pcClusterCenter(1) += pcCluster[i](1)/numPoints;
            pcClusterCenter(2) += pcCluster[i](2)/numPoints;
        }

        // std
        for (int i=0 ; i<numPoints ; i++){
            pcClusterStd(0) += std::pow(pcCluster[i](0) - pcClusterCenter(0),2);
            pcClusterStd(1) += std::pow(pcCluster[i](1) - pcClusterCenter(1),2);
            pcClusterStd(2) += std::pow(pcCluster[i](2) - pcClusterCenter(2),2);
        }        

        // take square root
        pcClusterStd(0) = std::sqrt(pcClusterStd(0)/numPoints);
        pcClusterStd(1) = std::sqrt(pcClusterStd(1)/numPoints);
        pcClusterStd(2) = std::sqrt(pcClusterStd(2)/numPoints);
    }

    void dynamicDetector::transformUVBBoxes(std::vector<onboardDetector::box3D>& bboxes){
        bboxes.clear();
        for(size_t i = 0; i < this->uvDetector_->box3Ds.size(); ++i){
            onboardDetector::box3D bbox;
            double x = this->uvDetector_->box3Ds[i].x; 
            double y = this->uvDetector_->box3Ds[i].y;
            double z = this->uvDetector_->box3Ds[i].z;
            double xWidth = this->uvDetector_->box3Ds[i].x_width;
            double yWidth = this->uvDetector_->box3Ds[i].y_width;
            double zWidth = this->uvDetector_->box3Ds[i].z_width;

            Eigen::Vector3d center (x, y, z);
            Eigen::Vector3d size (xWidth, yWidth, zWidth);
            Eigen::Vector3d newCenter, newSize;

            this->transformBBox(center, size, this->positionDepth_, this->orientationDepth_, newCenter, newSize);

            // assign values to bounding boxes in the map frame
            bbox.x = newCenter(0);
            bbox.y = newCenter(1);
            bbox.z = newCenter(2);
            bbox.x_width = newSize(0);
            bbox.y_width = newSize(1);
            bbox.z_width = newSize(2);
            bboxes.push_back(bbox);            
        }        
    }

    void dynamicDetector::boxAssociation(std::vector<int>& bestMatch, std::vector<int> &boxOOR){
        int numObjs = int(this->filteredBBoxes_.size());
        
        if (this->boxHist_.size() == 0){ // initialize new bounding box history if no history exists
            this->boxHist_.resize(numObjs);
            this->pcHist_.resize(numObjs);
            bestMatch.resize(this->filteredBBoxes_.size(), -1); // first detection no match
            for (int i=0 ; i<numObjs ; ++i){
                // initialize history for bbox, pc and KF
                this->boxHist_[i].push_back(this->filteredBBoxes_[i]);
                this->pcHist_[i].push_back(this->filteredPcClusters_[i]);
                MatrixXd states, A, B, H, P, Q, R;       
                this->kalmanFilterMatrixAcc(this->filteredBBoxes_[i], states, A, B, H, P, Q, R);
                onboardDetector::kalman_filter newFilter;
                newFilter.setup(states, A, B, H, P, Q, R);
                this->filters_.push_back(newFilter);
            }
        }
        else{
            // start association only if a new detection is available
            if (this->newDetectFlag_){
                this->boxAssociationHelper(bestMatch, boxOOR);
            }
        }

        this->newDetectFlag_ = false; // the most recent detection has been associated
    }

    void dynamicDetector::boxAssociationHelper(std::vector<int>& bestMatch, std::vector<int> &boxOOR){
        int numObjs = int(this->filteredBBoxes_.size());
        std::vector<onboardDetector::box3D> propedBoxes;
        std::vector<Eigen::VectorXd> propedBoxesFeat;
        std::vector<Eigen::VectorXd> currBoxesFeat;
        bestMatch.resize(numObjs);
        std::deque<std::deque<onboardDetector::box3D>> boxHistTemp; 

        // linear propagation: prediction of previous box in current frame
        this->linearProp(propedBoxes);

        // generate feature
        this->genFeat(propedBoxes, numObjs, propedBoxesFeat, currBoxesFeat);

        // calculate association: find best match
        this->findBestMatch(propedBoxesFeat, currBoxesFeat, propedBoxes, bestMatch);  
        if (this->boxHist_.size()){
            this->getBoxOutofRange(boxOOR, bestMatch);
            int numOORBox = 0;
            for (int i=0; i<int(boxOOR.size());i++){
                if (boxOOR[i]){
                    numOORBox++;
                }
            }
            if (numOORBox){
                this->findBestMatchEstimate(propedBoxesFeat, currBoxesFeat, propedBoxes, bestMatch, boxOOR); // TODO: is this useful?
            }
        }
    }

    void dynamicDetector::linearProp(std::vector<onboardDetector::box3D>& propedBoxes){
        onboardDetector::box3D propedBox;
        for (size_t i=0 ; i<this->boxHist_.size() ; i++){
            propedBox = this->boxHist_[i][0];
            propedBox.is_estimated = this->boxHist_[i][0].is_estimated;
            propedBox.x += propedBox.Vx*this->dt_;
            propedBox.y += propedBox.Vy*this->dt_;
            propedBoxes.push_back(propedBox);
        }
    }

    void dynamicDetector::genFeat(const std::vector<onboardDetector::box3D>& propedBoxes, int numObjs, std::vector<Eigen::VectorXd>& propedBoxesFeat, std::vector<Eigen::VectorXd>& currBoxesFeat){
        propedBoxesFeat.resize(propedBoxes.size());
        currBoxesFeat.resize(numObjs);
        this->genFeatHelper(propedBoxesFeat, propedBoxes);
        this->genFeatHelper(currBoxesFeat, this->filteredBBoxes_);
    }

    void dynamicDetector::genFeatHelper(std::vector<Eigen::VectorXd>& features, const std::vector<onboardDetector::box3D>& boxes){ 
        Eigen::VectorXd featureWeights(10); // 3pos + 3size + 1 pc length + 3 pc std
        featureWeights << 2, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5;
        for (size_t i=0 ; i<boxes.size() ; i++){
            Eigen::VectorXd feature(10);
            features[i] = feature;
            features[i](0) = (boxes[i].x - this->position_(0)) * featureWeights(0) ;
            features[i](1) = (boxes[i].y - this->position_(1)) * featureWeights(1);
            features[i](2) = (boxes[i].z - this->position_(2)) * featureWeights(2);
            features[i](3) = boxes[i].x_width * featureWeights(3);
            features[i](4) = boxes[i].y_width * featureWeights(4);
            features[i](5) = boxes[i].z_width * featureWeights(5);
            features[i](6) = this->filteredPcClusters_[i].size() * featureWeights(6);
            features[i](7) = this->filteredPcClusterStds_[i](0) * featureWeights(7);
            features[i](8) = this->filteredPcClusterStds_[i](1) * featureWeights(8);
            features[i](9) = this->filteredPcClusterStds_[i](2) * featureWeights(9);
        }
    }

    void dynamicDetector::findBestMatch(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch){
        int numObjs = this->filteredBBoxes_.size();
        std::vector<double> bestSims; // best similarity
        bestSims.resize(numObjs);

        for (int i=0 ; i<numObjs ; i++){
            double bestSim = -1.;
            int bestMatchInd = -1;
            for (size_t j=0 ; j<propedBoxes.size() ; j++){
                double sim = propedBoxesFeat[j].dot(currBoxesFeat[i])/(propedBoxesFeat[j].norm()*currBoxesFeat[i].norm());
                if (sim >= bestSim){
                    bestSim = sim;
                    bestSims[i] = sim;
                    bestMatchInd = j;
                }
            }

            double iou = this->calBoxIOU(this->filteredBBoxes_[i], propedBoxes[bestMatchInd]);
            if(!(bestSims[i]>this->simThresh_ && iou)){
                bestSims[i] = 0;
                bestMatch[i] = -1;
            }
            else {
                bestMatch[i] = bestMatchInd;
            }
        }
    }

    void dynamicDetector::findBestMatchEstimate(const std::vector<Eigen::VectorXd>& propedBoxesFeat, const std::vector<Eigen::VectorXd>& currBoxesFeat, const std::vector<onboardDetector::box3D>& propedBoxes, std::vector<int>& bestMatch, std::vector<int>& boxOOR){
        int numObjs = int(this->filteredBBoxes_.size());
        std::vector<double> bestSims; // best similarity
        bestSims.resize(numObjs);

        for (int i=0 ; i<numObjs ; i++){
            if (bestMatch[i] < 0){
                double bestSim = -1.;
                int bestMatchInd = -1;
                for (size_t j=0 ; j<propedBoxes.size() ; j++){
                    if (propedBoxes[j].is_estimated and boxOOR[j]){
                        double sim = propedBoxesFeat[j].dot(currBoxesFeat[i])/(propedBoxesFeat[j].norm()*currBoxesFeat[i].norm());
                        if (sim >= bestSim){
                            bestSim = sim;
                            bestSims[i] = sim;
                            bestMatchInd = j;
                        }
                    }
                }
                double iou = this->calBoxIOU(this->filteredBBoxes_[i], propedBoxes[bestMatchInd]);
                if(!(bestSims[i]>this->simThreshRetrack_ && iou)){
                    bestSims[i] = 0;
                    bestMatch[i] = -1;
                }
                else {
                    boxOOR[bestMatchInd] = 0;
                    bestMatch[i] = bestMatchInd;
                    // cout<<"retrack"<<endl;
                }
            }
        }
    }    

    void dynamicDetector::getBoxOutofRange(std::vector<int>& boxOOR, const std::vector<int>&bestMatch){
        if (int(this->boxHist_.size())>0){
            boxOOR.resize(this->boxHist_.size(), 1);
            for (int i=0; i<int(bestMatch.size()); i++){
                if (bestMatch[i]>=0){
                    boxOOR[bestMatch[i]] = 0;
                }
            }
            for (int i=0; i<int(boxOOR.size()); i++){
                if (boxOOR[i] and this->boxHist_[i][0].is_dynamic){
                    // cout<<"dynamic obstacle out of range"<<endl;
                }
                else{
                    boxOOR[i] = 0;
                }
            }
        }     
    }

    int dynamicDetector::getEstimateFrameNum(const std::deque<onboardDetector::box3D> &boxHist){
        int frameNum = 0;
        if (boxHist.size()){
            for (int i=0; i<int(boxHist.size()); i++){
                if (boxHist[i].is_estimated){
                    frameNum++;
                }
                else{
                    break;
                }
            }
        }
        return frameNum;
    }

    void dynamicDetector::kalmanFilterAndUpdateHist(const std::vector<int>& bestMatch, const std::vector<int> &boxOOR){
        std::vector<std::deque<onboardDetector::box3D>> boxHistTemp; 
        std::vector<std::deque<std::vector<Eigen::Vector3d>>> pcHistTemp;
        std::vector<onboardDetector::kalman_filter> filtersTemp;
        std::deque<onboardDetector::box3D> newSingleBoxHist;
        std::deque<std::vector<Eigen::Vector3d>> newSinglePcHist; 
        onboardDetector::kalman_filter newFilter;
        std::vector<onboardDetector::box3D> trackedBBoxesTemp;

        newSingleBoxHist.resize(0);
        newSinglePcHist.resize(0);
        int numObjs = this->filteredBBoxes_.size();

        for (int i=0 ; i<numObjs ; i++){
            onboardDetector::box3D newEstimatedBBox; // from kalman filter

            // inheret history. push history one by one
            if (bestMatch[i]>=0){
                boxHistTemp.push_back(this->boxHist_[bestMatch[i]]);
                pcHistTemp.push_back(this->pcHist_[bestMatch[i]]);
                filtersTemp.push_back(this->filters_[bestMatch[i]]);

                // kalman filter to get new state estimation
                onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];

                Eigen::MatrixXd Z;
                this->getKalmanObservationAcc(currDetectedBBox, bestMatch[i], Z);
                filtersTemp.back().estimate(Z, MatrixXd::Zero(6,1));
                
                
                newEstimatedBBox.x = filtersTemp.back().output(0);
                newEstimatedBBox.y = filtersTemp.back().output(1);
                newEstimatedBBox.z = currDetectedBBox.z;
                newEstimatedBBox.Vx = filtersTemp.back().output(2);
                newEstimatedBBox.Vy = filtersTemp.back().output(3);
                newEstimatedBBox.Ax = filtersTemp.back().output(4);
                newEstimatedBBox.Ay = filtersTemp.back().output(5);   
                          

                newEstimatedBBox.x_width = currDetectedBBox.x_width;
                newEstimatedBBox.y_width = currDetectedBBox.y_width;
                newEstimatedBBox.z_width = currDetectedBBox.z_width;
                newEstimatedBBox.is_dynamic = currDetectedBBox.is_dynamic;
                newEstimatedBBox.is_human = currDetectedBBox.is_human;
                newEstimatedBBox.is_estimated = false;
            }
            else{
                boxHistTemp.push_back(newSingleBoxHist);
                pcHistTemp.push_back(newSinglePcHist);

                // create new kalman filter for this object
                onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];
                MatrixXd states, A, B, H, P, Q, R;    
                this->kalmanFilterMatrixAcc(currDetectedBBox, states, A, B, H, P, Q, R);
                
                newFilter.setup(states, A, B, H, P, Q, R);
                filtersTemp.push_back(newFilter);
                newEstimatedBBox = currDetectedBBox;
                
            }

            // pop old data if len of hist > size limit
            if (int(boxHistTemp[i].size()) == this->histSize_){
                boxHistTemp[i].pop_back();
                pcHistTemp[i].pop_back();
            }

            // push new data into history
            boxHistTemp[i].push_front(newEstimatedBBox); 
            pcHistTemp[i].push_front(this->filteredPcClusters_[i]);

            // update new tracked bounding boxes
            trackedBBoxesTemp.push_back(newEstimatedBBox);
        }
        if (boxOOR.size()){
            for (int i=0; i<int(boxOOR.size()); i++){
                onboardDetector::box3D newEstimatedBBox; // from kalman filter 
                if (boxOOR[i] and this->getEstimateFrameNum(this->boxHist_[i]) < min(this->predSize_,this->histSize_-1)){
                    onboardDetector::box3D currDetectedBBox;
                    currDetectedBBox = this->boxHist_[i][0];
                    currDetectedBBox.x += this->dt_* currDetectedBBox.Vx;
                    currDetectedBBox.y += this->dt_* currDetectedBBox.Vy;

                    boxHistTemp.push_back(this->boxHist_[i]);
                    pcHistTemp.push_back(this->pcHist_[i]);
                    filtersTemp.push_back(this->filters_[i]);

                    Eigen::MatrixXd Z;
                    this->getKalmanObservationAcc(currDetectedBBox, i, Z);
                    filtersTemp.back().estimate(Z, MatrixXd::Zero(6,1));
                    
                    newEstimatedBBox.x = filtersTemp.back().output(0);
                    newEstimatedBBox.y = filtersTemp.back().output(1);
                    newEstimatedBBox.z = currDetectedBBox.z;
                    newEstimatedBBox.Vx = filtersTemp.back().output(2);
                    newEstimatedBBox.Vy = filtersTemp.back().output(3);
                    newEstimatedBBox.Ax = filtersTemp.back().output(4);
                    newEstimatedBBox.Ay = filtersTemp.back().output(5);   
                            
                    
                    newEstimatedBBox.x_width = currDetectedBBox.x_width;
                    newEstimatedBBox.y_width = currDetectedBBox.y_width;
                    newEstimatedBBox.z_width = currDetectedBBox.z_width;
                    newEstimatedBBox.is_dynamic = true;
                    newEstimatedBBox.is_human = currDetectedBBox.is_human;
                    newEstimatedBBox.is_estimated = true;
                    newEstimatedBBox.is_dynamic_candidate = true;

                    // pop old data if len of hist > size limit
                    if (int(boxHistTemp.back().size()) == this->histSize_){
                        boxHistTemp.back().pop_back();
                        pcHistTemp.back().pop_back();
                    }

                    // push new data into history
                    boxHistTemp.back().push_front(newEstimatedBBox);
                    pcHistTemp.back().push_front(this->pcHist_[i][0]);
                    trackedBBoxesTemp.push_back(newEstimatedBBox);
                }
            }
        }

        if (boxHistTemp.size()){
            for (size_t i=0; i<trackedBBoxesTemp.size(); ++i){ 
                if (not boxHistTemp[i][0].is_estimated){
                    if (int(boxHistTemp[i].size()) >= this->fixSizeHistThresh_){
                        if ((abs(trackedBBoxesTemp[i].x_width-boxHistTemp[i][1].x_width)/boxHistTemp[i][1].x_width) <= this->fixSizeDimThresh_ &&
                            (abs(trackedBBoxesTemp[i].y_width-boxHistTemp[i][1].y_width)/boxHistTemp[i][1].y_width) <= this->fixSizeDimThresh_&&
                            (abs(trackedBBoxesTemp[i].z_width-boxHistTemp[i][1].z_width)/boxHistTemp[i][1].z_width) <= this->fixSizeDimThresh_){
                            trackedBBoxesTemp[i].x_width = boxHistTemp[i][1].x_width;
                            trackedBBoxesTemp[i].y_width = boxHistTemp[i][1].y_width;
                            trackedBBoxesTemp[i].z_width = boxHistTemp[i][1].z_width;
                            boxHistTemp[i][0].x_width = trackedBBoxesTemp[i].x_width;
                            boxHistTemp[i][0].y_width = trackedBBoxesTemp[i].y_width;
                            boxHistTemp[i][0].z_width = trackedBBoxesTemp[i].z_width;
                        }

                    }
                }
            }
        }
        
        // update history member variable
        this->boxHist_ = boxHistTemp;
        this->pcHist_ = pcHistTemp;
        this->filters_ = filtersTemp;

        // update tracked bounding boxes
        this->trackedBBoxes_=  trackedBBoxesTemp;
    }

    void dynamicDetector::kalmanFilterMatrixAcc(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R){
        states.resize(6,1);
        states(0) = currDetectedBBox.x;
        states(1) = currDetectedBBox.y;
        // init vel and acc to zeros
        states(2) = 0.;
        states(3) = 0.;
        states(4) = 0.;
        states(5) = 0.;

        MatrixXd ATemp;
        ATemp.resize(6, 6);

        ATemp <<  1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2), 0,
                  0, 1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2),
                  0, 0, 1, 0, this->dt_, 0,
                  0 ,0, 0, 1, 0, this->dt_,
                  0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1;
        A = ATemp;
        B = MatrixXd::Zero(6, 6);
        H = MatrixXd::Identity(6, 6);
        P = MatrixXd::Identity(6, 6) * this->eP_;
        Q = MatrixXd::Identity(6, 6);
        Q(0,0) *= this->eQPos_; Q(1,1) *= this->eQPos_; Q(2,2) *= this->eQVel_; Q(3,3) *= this->eQVel_; Q(4,4) *= this->eQAcc_; Q(5,5) *= this->eQAcc_;
        R = MatrixXd::Identity(6, 6);
        R(0,0) *= this->eRPos_; R(1,1) *= this->eRPos_; R(2,2) *= this->eRVel_; R(3,3) *= this->eRVel_; R(4,4) *= this->eRAcc_; R(5,5) *= this->eRAcc_;
    }

    void dynamicDetector::getKalmanObservationAcc(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z){
        Z.resize(6, 1);
        Z(0) = currDetectedBBox.x;
        Z(1) = currDetectedBBox.y;

        // use previous k frame for velocity estimation
        int k = this->kfAvgFrames_;
        int historySize = this->boxHist_[bestMatchIdx].size();
        if (historySize < k){
            k = historySize;
        }
        onboardDetector::box3D prevMatchBBox = this->boxHist_[bestMatchIdx][k-1];

        Z(2) = (currDetectedBBox.x - prevMatchBBox.x)/(this->dt_*k);
        Z(3) = (currDetectedBBox.y - prevMatchBBox.y)/(this->dt_*k);
        Z(4) = (Z(2) - prevMatchBBox.Vx)/(this->dt_*k);
        Z(5) = (Z(3) - prevMatchBBox.Vy)/(this->dt_*k);
    }

    void dynamicDetector::getEstimateBox(const std::deque<onboardDetector::box3D> &boxHist, onboardDetector::box3D &estimatedBBox){
        onboardDetector::box3D lastDetect; lastDetect.x = 0; lastDetect.y = 0;
        for (int i=0; i<int(boxHist.size()); i++){
            if (not boxHist[i].is_estimated){
                lastDetect = boxHist[i];
                break;
            }
        }
        estimatedBBox.x = boxHist[0].x - (boxHist[0].x-lastDetect.x)/2;
        estimatedBBox.y = boxHist[0].y - (boxHist[0].y-lastDetect.y)/2;
        estimatedBBox.z = boxHist[0].z;
        estimatedBBox.x_width = std::min(std::abs(boxHist[0].x-lastDetect.x) + boxHist[0].x_width, 1.5 * boxHist[0].x_width);
        estimatedBBox.y_width = std::min(std::abs(boxHist[0].y-lastDetect.y) + boxHist[0].y_width, 1.5 * boxHist[0].y_width);
        estimatedBBox.z_width = boxHist[0].z_width;
        estimatedBBox.is_estimated = true;
    }

    // user functions
    void dynamicDetector::getDynamicObstacles(std::vector<onboardDetector::box3D>& incomeDynamicBBoxes, const Eigen::Vector3d &robotSize){
        incomeDynamicBBoxes.clear();
        for (int i=0; i<int(this->dynamicBBoxes_.size()); i++){
            onboardDetector::box3D box = this->dynamicBBoxes_[i];
            box.x_width += robotSize(0);
            box.y_width += robotSize(1);
            box.z_width += robotSize(2);
            incomeDynamicBBoxes.push_back(box);
        }
    }

    void dynamicDetector::getDynamicObstaclesHist(std::vector<std::vector<Eigen::Vector3d>>& posHist, std::vector<std::vector<Eigen::Vector3d>>& velHist, std::vector<std::vector<Eigen::Vector3d>>& sizeHist, const Eigen::Vector3d &robotSize){
		posHist.clear();
        velHist.clear();
        sizeHist.clear();

        if (this->boxHist_.size()){
            for (size_t i=0 ; i<this->boxHist_.size() ; ++i){
                if (this->boxHist_[i][0].is_dynamic or this->boxHist_[i][0].is_human){   
                    bool findMatch = false;     
                    if (this->constrainSize_){
                        for (Eigen::Vector3d targetSize : this->targetObjectSize_){
                            double xdiff = std::abs(this->boxHist_[i][0].x_width - targetSize(0));
                            double ydiff = std::abs(this->boxHist_[i][0].y_width - targetSize(1));
                            double zdiff = std::abs(this->boxHist_[i][0].z_width - targetSize(2)); 
                            if (xdiff < 0.8 and ydiff < 0.8 and zdiff < 1.0){
                                findMatch = true;
                            }
                        }
                    }
                    else{
                        findMatch = true;
                    }
                    if (findMatch){
                        std::vector<Eigen::Vector3d> obPosHist, obVelHist, obSizeHist;
                        for (size_t j=0; j<this->boxHist_[i].size() ; ++j){
                            Eigen::Vector3d pos(this->boxHist_[i][j].x, this->boxHist_[i][j].y, this->boxHist_[i][j].z);
                            Eigen::Vector3d vel(this->boxHist_[i][j].Vx, this->boxHist_[i][j].Vy, 0);
                            Eigen::Vector3d size(this->boxHist_[i][j].x_width, this->boxHist_[i][j].y_width, this->boxHist_[i][j].z_width);
                            size += robotSize;
                            obPosHist.push_back(pos);
                            obVelHist.push_back(vel);
                            obSizeHist.push_back(size);
                        }
                        posHist.push_back(obPosHist);
                        velHist.push_back(obVelHist);
                        sizeHist.push_back(obSizeHist);
                    }
                }
            }
        }
	}

    void dynamicDetector::getDynamicPc(std::vector<Eigen::Vector3d>& dynamicPc){
        Eigen::Vector3d curPoint;
        for (size_t i=0 ; i<this->filteredPoints_.size() ; ++i){
            curPoint = this->filteredPoints_[i];
            for (size_t j=0; j<this->dynamicBBoxes_.size() ; ++j){
                if (abs(curPoint(0)-this->dynamicBBoxes_[j].x)<=this->dynamicBBoxes_[j].x_width/2 and 
                    abs(curPoint(1)-this->dynamicBBoxes_[j].y)<=this->dynamicBBoxes_[j].y_width/2 and 
                    abs(curPoint(2)-this->dynamicBBoxes_[j].z)<=this->dynamicBBoxes_[j].z_width/2) {
                        dynamicPc.push_back(curPoint);
                        break;
                    }
            }
        }
    } 
    
    void dynamicDetector::publishUVImages(){
        sensor_msgs::msg::Image::SharedPtr depthBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->depth_show).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr umapBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->U_map_show).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr birdBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->bird_view).toImageMsg();  
        this->uvDepthMapPub_->publish(*depthBoxMsg);
        this->uDepthMapPub_->publish(*umapBoxMsg); 
        this->uvBirdViewPub_->publish(*birdBoxMsg);     
    }

    void dynamicDetector::publishColorImages(){
        sensor_msgs::msg::Image::SharedPtr detectedColorImgMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", this->detectedColorImage_).toImageMsg();
        this->detectedColorImgPub_->publish(*detectedColorImgMsg);
    }

    void dynamicDetector::publishPoints(const std::vector<Eigen::Vector3d>& points, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher){
        pcl::PointXYZ pt;
        pcl::PointCloud<pcl::PointXYZ> cloud;        
        for (size_t i=0; i<points.size(); ++i){
            pt.x = points[i](0);
            pt.y = points[i](1);
            pt.z = points[i](2);
            cloud.push_back(pt);
        }    
        cloud.width = cloud.points.size();
        cloud.height = 1;
        cloud.is_dense = true;
        cloud.header.frame_id = "map";

        sensor_msgs::msg::PointCloud2 cloudMsg;
        pcl::toROSMsg(cloud, cloudMsg);
        publisher->publish(cloudMsg);
    }


    void dynamicDetector::publish3dBox(const std::vector<box3D>& boxes, const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher, double r, double g, double b) {
        // visualization using bounding boxes 
        visualization_msgs::msg::Marker line;
        visualization_msgs::msg::MarkerArray lines;
        line.header.frame_id = "map";
        line.type = visualization_msgs::msg::Marker::LINE_LIST;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.ns = "box3D";  
        line.scale.x = 0.06;
        line.color.r = r;
        line.color.g = g;
        line.color.b = b;
        line.color.a = 1.0;
        line.lifetime = rclcpp::Duration::from_seconds(0.1);
        
        for(size_t i = 0; i < boxes.size(); i++){
            // for estimated bbox, using a different color
            if (boxes[i].is_estimated){
                 line.color.r = 0.8;
                 line.color.g = 0.2;
                 line.color.b = 0.0;
                 line.color.a = 1.0;
            }

            // visualization msgs
            line.text = " Vx " + std::to_string(boxes[i].Vx) + " Vy " + std::to_string(boxes[i].Vy);
            double x = boxes[i].x; 
            double y = boxes[i].y; 
            double z = (boxes[i].z+boxes[i].z_width/2)/2; 

            // double x_width = std::max(boxes[i].x_width,boxes[i].y_width);
            // double y_width = std::max(boxes[i].x_width,boxes[i].y_width);
            double x_width = boxes[i].x_width;
            double y_width = boxes[i].y_width;
            double z_width = 2*z;

            // double z = 
            
            vector<geometry_msgs::msg::Point> verts;
            geometry_msgs::msg::Point p;
            // vertice 0
            p.x = x-x_width / 2.; p.y = y-y_width / 2.; p.z = z-z_width / 2.;
            verts.push_back(p);

            // vertice 1
            p.x = x-x_width / 2.; p.y = y+y_width / 2.; p.z = z-z_width / 2.;
            verts.push_back(p);

            // vertice 2
            p.x = x+x_width / 2.; p.y = y+y_width / 2.; p.z = z-z_width / 2.;
            verts.push_back(p);

            // vertice 3
            p.x = x+x_width / 2.; p.y = y-y_width / 2.; p.z = z-z_width / 2.;
            verts.push_back(p);

            // vertice 4
            p.x = x-x_width / 2.; p.y = y-y_width / 2.; p.z = z+z_width / 2.;
            verts.push_back(p);

            // vertice 5
            p.x = x-x_width / 2.; p.y = y+y_width / 2.; p.z = z+z_width / 2.;
            verts.push_back(p);

            // vertice 6
            p.x = x+x_width / 2.; p.y = y+y_width / 2.; p.z = z+z_width / 2.;
            verts.push_back(p);

            // vertice 7
            p.x = x+x_width / 2.; p.y = y-y_width / 2.; p.z = z+z_width / 2.;
            verts.push_back(p);
            
            int vert_idx[12][2] = {
                {0,1},
                {1,2},
                {2,3},
                {0,3},
                {0,4},
                {1,5},
                {3,7},
                {2,6},
                {4,5},
                {5,6},
                {4,7},
                {6,7}
            };
            
            for (size_t i=0;i<12;i++){
                line.points.push_back(verts[vert_idx[i][0]]);
                line.points.push_back(verts[vert_idx[i][1]]);
            }
            
            lines.markers.push_back(line);
            
            line.id++;
        }
        // publish
        publisher->publish(lines);
    }

    void dynamicDetector::publishHistoryTraj(){
        visualization_msgs::msg::MarkerArray trajMsg;
        int countMarker = 0;
        for (size_t i=0; i<this->boxHist_.size(); ++i){
            visualization_msgs::msg::Marker traj;
            traj.header.frame_id = "map";
            traj.header.stamp = this->get_clock()->now();
            traj.ns = "dynamic_detector";
            traj.id = countMarker;
            traj.type = visualization_msgs::msg::Marker::LINE_LIST;
            traj.lifetime = rclcpp::Duration::from_seconds(0.1);
            traj.scale.x = 0.03;
            traj.scale.y = 0.03;
            traj.scale.z = 0.03;
            traj.color.a = 1.0; // Don't forget to set the alpha!
            traj.color.r = 0.0;
            traj.color.g = 1.0;
            traj.color.b = 0.0;
            for (size_t j=0; j<this->boxHist_[i].size()-1; ++j){
                geometry_msgs::msg::Point p1, p2;
                onboardDetector::box3D box1 = this->boxHist_[i][j];
                onboardDetector::box3D box2 = this->boxHist_[i][j+1];
                p1.x = box1.x; p1.y = box1.y; p1.z = box1.z;
                p2.x = box2.x; p2.y = box2.y; p2.z = box2.z;
                traj.points.push_back(p1);
                traj.points.push_back(p2);
            }

            ++countMarker;
            trajMsg.markers.push_back(traj);
        }
        this->historyTrajPub_->publish(trajMsg);
    }

    void dynamicDetector::publishVelVis(){ // publish velocities for all tracked objects
        visualization_msgs::msg::MarkerArray velVisMsg;
        int countMarker = 0;
        for (size_t i=0; i<this->trackedBBoxes_.size(); ++i){
            visualization_msgs::msg::Marker velMarker;
            velMarker.header.frame_id = "map";
            velMarker.header.stamp = this->get_clock()->now();
            velMarker.ns = "dynamic_detector";
            velMarker.id =  countMarker;
            velMarker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            velMarker.pose.position.x = this->trackedBBoxes_[i].x;
            velMarker.pose.position.y = this->trackedBBoxes_[i].y;
            velMarker.pose.position.z = this->trackedBBoxes_[i].z + this->trackedBBoxes_[i].z_width/2. + 0.3;
            velMarker.scale.x = 0.15;
            velMarker.scale.y = 0.15;
            velMarker.scale.z = 0.15;
            velMarker.color.a = 1.0;
            velMarker.color.r = 1.0;
            velMarker.color.g = 0.0;
            velMarker.color.b = 0.0;
            velMarker.lifetime = rclcpp::Duration::from_seconds(0.1);
            double vx = this->trackedBBoxes_[i].Vx;
            double vy = this->trackedBBoxes_[i].Vy;
            double vNorm = sqrt(vx*vx+vy*vy);

            // Format the text with two decimal places
            std::ostringstream velStream;
            velStream << std::fixed << std::setprecision(2);
            velStream << "Vx=" << vx << ", Vy=" << vy << ", |V|=" << vNorm;
            velMarker.text = velStream.str();
            velVisMsg.markers.push_back(velMarker);
            ++countMarker;
        }
        this->velVisPub_->publish(velVisMsg);
    }

    void dynamicDetector::updatePoseHist(){ 
        if (int(this->positionHist_.size()) == this->skipFrame_){
            this->positionHist_.pop_back();
        }
        else{
            this->positionHist_.push_front(this->position_);
        }
        if (int(this->orientationHist_.size()) == this->skipFrame_){
            this->orientationHist_.pop_back();
        }
        else{
            this->orientationHist_.push_front(this->orientation_);
        }
    }

    void dynamicDetector::transformBBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size, const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation,
                                               Eigen::Vector3d& newCenter, Eigen::Vector3d& newSize){
        double x = center(0); 
        double y = center(1);
        double z = center(2);
        double xWidth = size(0);
        double yWidth = size(1);
        double zWidth = size(2);

        // get 8 bouding boxes coordinates in the camera frame
        Eigen::Vector3d p1 (x+xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p2 (x+xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p3 (x+xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p4 (x+xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p5 (x-xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p6 (x-xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p7 (x-xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p8 (x-xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);

        // transform 8 points to the map coordinate frame
        Eigen::Vector3d p1m = orientation * p1 + position;
        Eigen::Vector3d p2m = orientation * p2 + position;
        Eigen::Vector3d p3m = orientation * p3 + position;
        Eigen::Vector3d p4m = orientation * p4 + position;
        Eigen::Vector3d p5m = orientation * p5 + position;
        Eigen::Vector3d p6m = orientation * p6 + position;
        Eigen::Vector3d p7m = orientation * p7 + position;
        Eigen::Vector3d p8m = orientation * p8 + position;
        std::vector<Eigen::Vector3d> pointsMap {p1m, p2m, p3m, p4m, p5m, p6m, p7m, p8m};

        // find max min in x, y, z directions
        double xmin=p1m(0); double xmax=p1m(0); 
        double ymin=p1m(1); double ymax=p1m(1);
        double zmin=p1m(2); double zmax=p1m(2);
        for (Eigen::Vector3d pm : pointsMap){
            if (pm(0) < xmin){xmin = pm(0);}
            if (pm(0) > xmax){xmax = pm(0);}
            if (pm(1) < ymin){ymin = pm(1);}
            if (pm(1) > ymax){ymax = pm(1);}
            if (pm(2) < zmin){zmin = pm(2);}
            if (pm(2) > zmax){zmax = pm(2);}
        }
        newCenter(0) = (xmin + xmax)/2.0;
        newCenter(1) = (ymin + ymax)/2.0;
        newCenter(2) = (zmin + zmax)/2.0;
        newSize(0) = xmax - xmin;
        newSize(1) = ymax - ymin;
        newSize(2) = zmax - zmin;
    }

    bool dynamicDetector::isInFov(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, Eigen::Vector3d& point){
        Eigen::Vector3d worldRay = point - position;
        Eigen::Vector3d camUnitX(1,0,0);
        Eigen::Vector3d camUnitY(0,1,0);
        Eigen::Vector3d camUnitZ(0,0,1);
        Eigen::Vector3d camRay;
        Eigen::Vector3d displacement; 
    
        // z is in depth direction in camera coord
        camRay = orientation.inverse()*worldRay;
        double camRayX = abs(camRay.dot(camUnitX));
        double camRayY = abs(camRay.dot(camUnitY));
        double camRayZ = abs(camRay.dot(camUnitZ));

        double htan = camRayX/camRayZ;
        double vtan = camRayY/camRayZ;
        
        double pi = 3.1415926;
        return htan<tan(42*pi/180) && vtan<tan(28*pi/180) && camRayZ<this->depthMaxValue_;
    }

    int dynamicDetector::getBestOverlapBBox(const onboardDetector::box3D& currBBox, const std::vector<onboardDetector::box3D>& targetBBoxes, double& bestIOU){
        bestIOU = 0.0;
        int bestIOUIdx = -1; // no match
        for (size_t i=0; i<targetBBoxes.size(); ++i){
            onboardDetector::box3D targetBBox = targetBBoxes[i];
            double IOU = this->calBoxIOU(currBBox, targetBBox);
            if (IOU > bestIOU){
                bestIOU = IOU;
                bestIOUIdx = i;
            }
        }
        return bestIOUIdx;
    }

    double dynamicDetector::calBoxIOU(const onboardDetector::box3D& box1, const onboardDetector::box3D& box2){
        double box1Volume = box1.x_width * box1.y_width * box1.z_width;
        double box2Volume = box2.x_width * box2.y_width * box2.z_width;

        double l1Y = box1.y+box1.y_width/2-(box2.y-box2.y_width/2);
        double l2Y = box2.y+box2.y_width/2-(box1.y-box1.y_width/2);
        double l1X = box1.x+box1.x_width/2-(box2.x-box2.x_width/2);
        double l2X = box2.x+box2.x_width/2-(box1.x-box1.x_width/2);
        double l1Z = box1.z+box1.z_width/2-(box2.z-box2.z_width/2);
        double l2Z = box2.z+box2.z_width/2-(box1.z-box1.z_width/2);
        double overlapX = std::min( l1X , l2X );
        double overlapY = std::min( l1Y , l2Y );
        double overlapZ = std::min( l1Z , l2Z );
       
        if (std::max(l1X, l2X)<=std::max(box1.x_width,box2.x_width)){ 
            overlapX = std::min(box1.x_width, box2.x_width);
        }
        if (std::max(l1Y, l2Y)<=std::max(box1.y_width,box2.y_width)){ 
            overlapY = std::min(box1.y_width, box2.y_width);
        }
        if (std::max(l1Z, l2Z)<=std::max(box1.z_width,box2.z_width)){ 
            overlapZ = std::min(box1.z_width, box2.z_width);
        }


        double overlapVolume = overlapX * overlapY *  overlapZ;
        double IOU = overlapVolume / (box1Volume+box2Volume-overlapVolume);
        
        // D-IOU
        if (overlapX<=0 || overlapY<=0 ||overlapZ<=0){
            IOU = 0;
        }
        return IOU;
    }
}