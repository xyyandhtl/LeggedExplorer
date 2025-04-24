/*
	FILE: occupancyMap.cpp
	--------------------------------------
	function definition of occupancy map
*/
#include <map_manager/occupancyMap.h>

namespace mapManager{
    occMap::occMap() : Node("map_manager_node"){
    }

	void occMap::initMap(){
		this->ns_ = "occupancy_map";
		this->hint_ = "[OccMap]";
		this->initParam();
		this->initPrebuiltMap();
		this->registerPub();
		this->registerCallback();
	}

	void occMap::initParam(){		
		// localization mode
		this->declare_parameter<int>("localization_mode", 0);
		this->get_parameter("localization_mode", this->localizationMode_);
		std::cout << this->hint_ << ": Localization mode: pose (0)/odom (1). Your option: " << this->localizationMode_ << std::endl;

		// depth topic name
		this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_raw");
		this->get_parameter("depth_image_topic", this->depthTopicName_);
		std::cout << this->hint_ << ": Depth topic: " << this->depthTopicName_ << std::endl;

		// pointcloud topic name
		this->declare_parameter<std::string>("point_cloud_topic", "/camera/depth/points");
		this->get_parameter("point_cloud_topic", this->pointcloudTopicName_);
		std::cout << this->hint_ << ": Pointcloud topic: " << this->pointcloudTopicName_ << std::endl;

		// conditional logic for localization mode
		if (this->localizationMode_ == 0) {
			// pose topic name
			this->declare_parameter<std::string>("pose_topic", "/CERLAB/quadcopter/pose");
			this->get_parameter("pose_topic", this->poseTopicName_);
			std::cout << this->hint_ << ": Pose topic: " << this->poseTopicName_ << std::endl;
		}else{
			// odom topic name
			this->declare_parameter<std::string>("odom_topic", "/CERLAB/quadcopter/odom");
			this->get_parameter("odom_topic", this->odomTopicName_);
			std::cout << this->hint_ << ": Odom topic: " << this->odomTopicName_ << std::endl;
		}

		// robot size
		this->declare_parameter<std::vector<double>>("robot_size", {0.5, 0.5, 0.3});
		std::vector<double> robotSizeVec(3);
		this->get_parameter("robot_size", robotSizeVec);
		std::cout << this->hint_ << ": Robot size: [" << robotSizeVec[0] << ", " << robotSizeVec[1] << ", " << robotSizeVec[2] << "]" << std::endl;
		this->robotSize_(0) = robotSizeVec[0]; this->robotSize_(1) = robotSizeVec[1]; this->robotSize_(2) = robotSizeVec[2];

		// depth intrinsics
		this->declare_parameter<std::vector<double>>("depth_intrinsics", {0., 0., 0., 0.});
		std::vector<double> depthIntrinsics(4);
		this->get_parameter("depth_intrinsics", depthIntrinsics);
		this->fx_ = depthIntrinsics[0];
		this->fy_ = depthIntrinsics[1];
		this->cx_ = depthIntrinsics[2];
		this->cy_ = depthIntrinsics[3];
		std::cout << this->hint_ << ": fx, fy, cx, cy: [" << this->fx_ << ", " << this->fy_ << ", " << this->cx_ << ", " << this->cy_ << "]" << std::endl;
		
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

		// transform matrix: body to depth sensor
		this->declare_parameter<std::vector<double>>("body_to_depth_sensor", std::vector<double>(16));
		std::vector<double> body2DepthSensorVec(16);
		this->get_parameter("body_to_depth_sensor", body2DepthSensorVec);
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				this->body2DepthSensor_(i, j) = body2DepthSensorVec[i * 4 + j];
			}
		}

		// pointcloud min distance
		this->declare_parameter<double>("pointcloud_min_distance", 0.2);
		this->get_parameter("pointcloud_min_distance", this->pointcloudMinDistance_);
		std::cout << this->hint_ << ": Pointcloud min distance: " << this->pointcloudMinDistance_ << std::endl;
		
		// pointcloud max distance
		this->declare_parameter<double>("pointcloud_max_distance", 5.0);
		this->get_parameter("pointcloud_max_distance", this->pointcloudMaxDistance_);
		std::cout << this->hint_ << ": Pointcloud max distance: " << this->pointcloudMaxDistance_ << std::endl;

		// transform matrix: body to point cloud sensor
		this->declare_parameter<std::vector<double>>("body_to_pointcloud_sensor", std::vector<double>(16));
		std::vector<double> body2PointcloudSensorVec(16);
		this->get_parameter("body_to_pointcloud_sensor", body2PointcloudSensorVec);
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				this->body2PointcloudSensor_(i, j) = body2PointcloudSensorVec[i * 4 + j];
			}
		}
		
		// Raycast max length
		this->declare_parameter<double>("raycast_max_length", 5.0);
		this->get_parameter("raycast_max_length", this->raycastMaxLength_);
		std::cout << this->hint_ << ": Raycast max length: " << this->raycastMaxLength_ << std::endl;

		// p hit
		this->declare_parameter<double>("p_hit", 0.70);
		double pHit;
		this->get_parameter("p_hit", pHit);
		std::cout << this->hint_ << ": P hit: " << pHit << std::endl;
		this->pHitLog_ = this->logit(pHit);

		// p miss
		this->declare_parameter<double>("p_miss", 0.35);
		double pMiss;
		this->get_parameter("p_miss", pMiss);
		std::cout << this->hint_ << ": P miss: " << pMiss << std::endl;
		this->pMissLog_ = this->logit(pMiss);

		// p min
		this->declare_parameter<double>("p_min", 0.12);
		double pMin;
		this->get_parameter("p_min", pMin);
		std::cout << this->hint_ << ": P min: " << pMin << std::endl;
		this->pMinLog_ = this->logit(pMin);

		// p max
		this->declare_parameter<double>("p_max", 0.97);
		double pMax;
		this->get_parameter("p_max", pMax);
		std::cout << this->hint_ << ": P max: " << pMax << std::endl;
		this->pMaxLog_ = this->logit(pMax);

		// p occ
		this->declare_parameter<double>("p_occ", 0.80);
		double pOcc;
		this->get_parameter("p_occ", pOcc);
		std::cout << this->hint_ << ": P occ: " << pOcc << std::endl;
		this->pOccLog_ = this->logit(pOcc);

		// map resolution
		this->declare_parameter<double>("map_resolution", 0.1);
		this->get_parameter("map_resolution", this->mapRes_);
		std::cout << this->hint_ << ": Map resolution: " << this->mapRes_ << std::endl;

		// ground height
		this->declare_parameter<double>("ground_height", 0.0);
		this->get_parameter("ground_height", this->groundHeight_);
		std::cout << this->hint_ << ": Ground height: " << this->groundHeight_ << std::endl;

		// map size
		this->declare_parameter<std::vector<double>>("map_size", {20.0, 20.0, 3.0});
		std::vector<double> mapSizeVec(3);
		this->get_parameter("map_size", mapSizeVec);
		std::cout << this->hint_ << ": Map size: [" << mapSizeVec[0] << ", " << mapSizeVec[1] << ", " << mapSizeVec[2] << "]" << std::endl;
		this->mapSize_(0) = mapSizeVec[0];
		this->mapSize_(1) = mapSizeVec[1];
		this->mapSize_(2) = mapSizeVec[2];

		// Initialize min/max for map boundaries
		this->mapSizeMin_(0) = -mapSizeVec[0] / 2;
		this->mapSizeMax_(0) = mapSizeVec[0] / 2;
		this->mapSizeMin_(1) = -mapSizeVec[1] / 2;
		this->mapSizeMax_(1) = mapSizeVec[1] / 2;
		this->mapSizeMin_(2) = this->groundHeight_;
		this->mapSizeMax_(2) = this->groundHeight_ + mapSizeVec[2];

		// min max for voxel
		this->mapVoxelMin_(0) = 0;
		this->mapVoxelMax_(0) = ceil(mapSizeVec[0] / this->mapRes_);
		this->mapVoxelMin_(1) = 0;
		this->mapVoxelMax_(1) = ceil(mapSizeVec[1] / this->mapRes_);
		this->mapVoxelMin_(2) = 0;
		this->mapVoxelMax_(2) = ceil(mapSizeVec[2] / this->mapRes_);

		// reserve vector for variables
		int reservedSize = this->mapVoxelMax_(0) * this->mapVoxelMax_(1) * this->mapVoxelMax_(2);
		this->countHitMiss_.resize(reservedSize, 0);
		this->countHit_.resize(reservedSize, 0);
		this->occupancy_.resize(reservedSize, this->pMinLog_ - this->UNKNOWN_FLAG_);
		this->occupancyInflated_.resize(reservedSize, false);
		this->flagTraverse_.resize(reservedSize, -1);
		this->flagRayend_.resize(reservedSize, -1);

		// local update range
		this->declare_parameter<std::vector<double>>("local_update_range", {5.0, 5.0, 3.0});
		std::vector<double> localUpdateRangeVec;
		this->get_parameter("local_update_range", localUpdateRangeVec);
		std::cout << this->hint_ << ": Local update range: [" << localUpdateRangeVec[0] << ", " << localUpdateRangeVec[1] << ", " << localUpdateRangeVec[2] << "]" << std::endl;
		this->localUpdateRange_(0) = localUpdateRangeVec[0];
		this->localUpdateRange_(1) = localUpdateRangeVec[1];
		this->localUpdateRange_(2) = localUpdateRangeVec[2];

		// local bound inflate factor
		this->declare_parameter<double>("local_bound_inflation", 0.0);
		this->get_parameter("local_bound_inflation", this->localBoundInflate_);
		std::cout << this->hint_ << ": Local bound inflate: " << this->localBoundInflate_ << std::endl;

		// whether to clean local map
		this->declare_parameter<bool>("clean_local_map", true);
		this->get_parameter("clean_local_map", this->cleanLocalMap_);
		std::cout << this->hint_ << ": Clean local map option is set to: " << this->cleanLocalMap_ << std::endl;

		// absolute dir of prebuilt map file (.pcd)
		this->declare_parameter<std::string>("prebuilt_map_directory", "");
		this->get_parameter("prebuilt_map_directory", this->prebuiltMapDir_);
		std::cout << this->hint_ << ": The prebuilt map absolute dir is found: " << this->prebuiltMapDir_ << std::endl;

		// local map size (visualization)
		this->declare_parameter<std::vector<double>>("local_map_size", {10.0, 10.0, 2.0});
		std::vector<double> localMapSizeVec;
		this->get_parameter("local_map_size", localMapSizeVec);
		std::cout << this->hint_ << ": Local map size: [" << localMapSizeVec[0] << ", " << localMapSizeVec[1] << ", " << localMapSizeVec[2] << "]" << std::endl;
		this->localMapSize_(0) = localMapSizeVec[0] / 2;
		this->localMapSize_(1) = localMapSizeVec[1] / 2;
		this->localMapSize_(2) = localMapSizeVec[2] / 2;
		this->localMapVoxel_(0) = int(ceil(localMapSizeVec[0] / (2 * this->mapRes_)));
		this->localMapVoxel_(1) = int(ceil(localMapSizeVec[1] / (2 * this->mapRes_)));
		this->localMapVoxel_(2) = int(ceil(localMapSizeVec[2] / (2 * this->mapRes_)));

		// max vis height
		this->declare_parameter<double>("max_height_visualization", 3.0);
		this->get_parameter("max_height_visualization", this->maxVisHeight_);
		std::cout << this->hint_ << ": Max visualization height: " << this->maxVisHeight_ << std::endl;

		// visualize global map
		this->declare_parameter<bool>("visualize_global_map", false);
		this->get_parameter("visualize_global_map", this->visGlobalMap_);
		std::cout << this->hint_ << ": Visualize map option. local (0)/global (1): " << this->visGlobalMap_ << std::endl;

		// verbose
		this->declare_parameter<bool>("verbose", true);
		this->get_parameter("verbose", this->verbose_);
		if (!this->verbose_) {
			std::cout << this->hint_ << ": Not display messages" << std::endl;
		} else {
			std::cout << this->hint_ << ": Display messages" << std::endl;
		}
	}

	void occMap::initPrebuiltMap(){
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		try{
			pcl::io::loadPCDFile<pcl::PointXYZ> (this->prebuiltMapDir_, *cloud);
		}catch (const std::exception& e){
			cout << this->hint_ << ": No prebuilt map found/not using the prebuilt map." << endl;
			return;
		}

		if (cloud->width * cloud->height == 0){
			cout << this->hint_ << ": No prebuilt map found/not using the prebuilt map." << endl;
			return;
		}

		cout << this->hint_ << ": Map loaded with " << cloud->width * cloud->height << " data points. " << endl;
		int address;
		Eigen::Vector3i pointIndex;
		Eigen::Vector3d pointPos;
		Eigen::Vector3i inflateIndex;
		int inflateAddress;

		// update occupancy info
		int xInflateSize = ceil(this->robotSize_(0)/(2*this->mapRes_));
		int yInflateSize = ceil(this->robotSize_(1)/(2*this->mapRes_));
		int zInflateSize = ceil(this->robotSize_(2)/(2*this->mapRes_));

		Eigen::Vector3d currMapRangeMin (0.0, 0.0, 0.0);
		Eigen::Vector3d currMapRangeMax (0.0, 0.0, 0.0);

		const int  maxIndex = this->mapVoxelMax_(0) * this->mapVoxelMax_(1) * this->mapVoxelMax_(2);
		for (const auto& point: *cloud)
		{
			address = this->posToAddress(point.x, point.y, point.z);
			pointPos(0) = point.x; pointPos(1) = point.y; pointPos(2) = point.z;
			this->posToIndex(pointPos, pointIndex);

			this->occupancy_[address] = this->pMaxLog_;
			// update map range
			if (pointPos(0) < currMapRangeMin(0)){
				currMapRangeMin(0) = pointPos(0);
			}

			if (pointPos(0) > currMapRangeMax(0)){
				currMapRangeMax(0) = pointPos(0);
			}

			if (pointPos(1) < currMapRangeMin(1)){
				currMapRangeMin(1) = pointPos(1);
			}

			if (pointPos(1) > currMapRangeMax(1)){
				currMapRangeMax(1) = pointPos(1);
			}

			if (pointPos(2) < currMapRangeMin(2)){
				currMapRangeMin(2) = pointPos(2);
			}

			if (pointPos(2) > currMapRangeMax(2)){
				currMapRangeMax(2) = pointPos(2);
			}

			for (int ix=-xInflateSize; ix<=xInflateSize; ++ix){
				for (int iy=-yInflateSize; iy<=yInflateSize; ++iy){
					for (int iz=-zInflateSize; iz<=zInflateSize; ++iz){
						inflateIndex(0) = pointIndex(0) + ix;
						inflateIndex(1) = pointIndex(1) + iy;
						inflateIndex(2) = pointIndex(2) + iz;
						inflateAddress = this->indexToAddress(inflateIndex);
						if ((inflateAddress < 0) or (inflateAddress > maxIndex)){
							continue; // those points are not in the reserved map
						} 
						this->occupancyInflated_[inflateAddress] = true;
					}
				}
			}
		}
		this->currMapRangeMin_ = currMapRangeMin;
		this->currMapRangeMax_ = currMapRangeMax;
	}


	void occMap::registerCallback(){
		// callback groups
		this->srvClientGroup_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
		this->mapGroup_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
		this->visGroup_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

		// sensor subscriber
		this->depthSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, this->depthTopicName_);
		this->pointcloudSub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, this->pointcloudTopicName_);

		// sync subscriber
		if (this->localizationMode_ == 0){
			this->poseSub_ = std::make_shared<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>>(this, this->poseTopicName_);
			this->pointcloudPoseSync_  = std::make_shared<message_filters::Synchronizer<pointcloudPoseSync>>(pointcloudPoseSync(100), *this->pointcloudSub_, *this->poseSub_);
			this->pointcloudPoseSync_->registerCallback(std::bind(&occMap::pointcloudPoseCB, this, std::placeholders::_1, std::placeholders::_2));
			this->depthPoseSync_  = std::make_shared<message_filters::Synchronizer<depthPoseSync>>(depthPoseSync(100), *this->depthSub_, *this->poseSub_);
			this->depthPoseSync_->registerCallback(std::bind(&occMap::depthPoseCB, this, std::placeholders::_1, std::placeholders::_2));
		}
		else{
			this->odomSub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, this->odomTopicName_);
			this->pointcloudOdomSync_ = std::make_shared<message_filters::Synchronizer<pointcloudOdomSync>>(pointcloudOdomSync(100), *this->pointcloudSub_, *this->odomSub_);
			this->pointcloudOdomSync_->registerCallback(std::bind(&occMap::pointcloudOdomCB, this, std::placeholders::_1, std::placeholders::_2));
			this->depthOdomSync_ = std::make_shared<message_filters::Synchronizer<depthOdomSync>>(depthOdomSync(100), *this->depthSub_, *this->odomSub_);
			this->depthOdomSync_->registerCallback(std::bind(&occMap::depthOdomCB, this, std::placeholders::_1, std::placeholders::_2));
		}


		// occupancy update callback
		this->occTimer_ = this->create_wall_timer(50ms, std::bind(&occMap::updateOccupancyCB, this), this->mapGroup_);

		// map inflation callback
		this->inflateTimer_ = this->create_wall_timer(50ms, std::bind(&occMap::inflateMapCB, this), this->mapGroup_);

		// static obstacle clustering callback
		this->staticClusteringTimer_ = this->create_wall_timer(50ms, std::bind(&occMap::staticClusteringCB, this), this->mapGroup_);

		// visualization callback
		this->visTimer_= this->create_wall_timer(50ms, std::bind(&occMap::visCB, this), this->mapGroup_); 

		// publish service
		// collision check server
		this->collisionCheckServer_ = this->create_service<map_manager::srv::CheckPosCollision>(this->ns_ + "/check_pos_collision", std::bind(&occMap::checkCollisionSrv, this, std::placeholders::_1, std::placeholders::_2));

		// static obstacles server
		this->staticObstacleServer_ = this->create_service<map_manager::srv::GetStaticObstacles>(this->ns_ + "/get_static_obstacles", std::bind(&occMap::getStaticObstaclesSrv, this, std::placeholders::_1, std::placeholders::_2));				

		// raycast server
		this->raycastServer_ = this->create_service<map_manager::srv::RayCast>(this->ns_ + "/raycast", std::bind(&occMap::getRayCastSrv, this, std::placeholders::_1, std::placeholders::_2));		
		this->getDynamicObstaclesClient_ = this->create_client<onboard_detector::srv::GetDynamicObstacles>("/onboard_detector/get_dynamic_obstacles", rmw_qos_profile_services_default, this->srvClientGroup_);
	}

	void occMap::registerPub(){
		this->depthCloudPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->ns_ + "/depth_cloud", 10);
		this->mapVisPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->ns_ + "/voxel_map", 10);
		this->inflatedMapVisPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->ns_ + "/inflated_voxel_map", 10);
		this->map2DPub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(this->ns_ + "/occupancy_map_2D", 10);
		this->staticObstacleVisPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/static_obstacles", 10);
		this->raycastVisPub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->ns_ + "/raycast", 10);
	}

	bool occMap::checkCollisionSrv(const std::shared_ptr<map_manager::srv::CheckPosCollision::Request> req, const std::shared_ptr<map_manager::srv::CheckPosCollision::Response> res){
		if (req->inflated){
			res->occupied = this->isInflatedOccupied(Eigen::Vector3d (req->x, req->y, req->z));
		}
		else{
			res->occupied = this->isOccupied(Eigen::Vector3d (req->x, req->y, req->z));
		}

		return true;
	}

	bool occMap::getStaticObstaclesSrv(const std::shared_ptr<map_manager::srv::GetStaticObstacles::Request> req, const std::shared_ptr<map_manager::srv::GetStaticObstacles::Response> res){
		for (int i=0; i<int(this->refinedBBoxVertices_.size()); ++i){
			bboxVertex staticObstacle = this->refinedBBoxVertices_[i];
			
			geometry_msgs::msg::Vector3 pos;
			geometry_msgs::msg::Vector3 size;
			
			pos.x = staticObstacle.centroid(0);
			pos.y = staticObstacle.centroid(1);
			pos.z = staticObstacle.centroid(2);

			size.x = staticObstacle.dimension(0);
			size.y = staticObstacle.dimension(1);
			size.z = staticObstacle.dimension(2);
			
			res->position.push_back(pos);
			res->size.push_back(size);
			res->angle.push_back(staticObstacle.angle);
		}
		return true;
	}

	bool occMap::getRayCastSrv(const std::shared_ptr<map_manager::srv::RayCast::Request> req, const std::shared_ptr<map_manager::srv::RayCast::Response> res){
		double hres = req->hres * M_PI/180.0;
		int numHbeams = int(360/req->hres);
		double vres = double(((req->vfov_max - req->vfov_min)* M_PI/180.0)/(req->vbeams-1));
		double vStartAngle = req->vfov_min * M_PI/180.0;
		int numVbeams = req->vbeams;
		double range = req->range;
		Eigen::Vector3d start (req->position.x, req->position.y, req->position.z);

		visualization_msgs::msg::MarkerArray rayVisMsg;
		geometry_msgs::msg::Point startPoint;
		startPoint.x = req->position.x;
		startPoint.y = req->position.y;
		startPoint.z = req->position.z;
		int id = 0;
		double starthAngle = req->start_angle;
		for (int h=0; h<numHbeams; ++h){
			double hAngle = starthAngle + double(h) * hres;
			Eigen::Vector3d hdirection (cos(hAngle), sin(hAngle), 0.0); // horizontal direction 
			for (int v=0; v<numVbeams; ++v){
				// get hit points
				double vAngle = vStartAngle + double(v) * vres;
				double vup = tan(vAngle);
				Eigen::Vector3d direction = hdirection;
				direction(2) += vup;
				direction /= direction.norm();
				Eigen::Vector3d hitPoint;
				bool success = this->castRay(start, direction, hitPoint, range, true);
				if (not success){
					hitPoint = start + range * direction;
				}
				for (int i=0; i<3; ++i){
					res->points.push_back(hitPoint(i));
				}
				
				if (req->visualize){
					visualization_msgs::msg::Marker point;
					point.header.frame_id = "map";
					point.header.stamp = rclcpp::Clock().now();
					point.ns = "raycast_points";
					point.id = id;
					point.type = visualization_msgs::msg::Marker::SPHERE;
					point.action = visualization_msgs::msg::Marker::ADD;
					point.pose.position.x = hitPoint(0);
					point.pose.position.y = hitPoint(1);
					point.pose.position.z = hitPoint(2);
					point.scale.x = 0.1;
					point.scale.y = 0.1;
					point.scale.z = 0.1;
					point.color.a = 1.0;
					point.color.r = 1.0;
					point.lifetime = rclcpp::Duration::from_seconds(0.5);
					rayVisMsg.markers.push_back(point);

					visualization_msgs::msg::Marker line;	
					line.header.frame_id = "map";
					line.header.stamp = rclcpp::Clock().now();
					line.ns = "raycast_lines";
					line.id = id;
					line.type = visualization_msgs::msg::Marker::LINE_LIST;
					line.action = visualization_msgs::msg::Marker::ADD;
					
					geometry_msgs::msg::Point endPoint;
					endPoint.x = hitPoint(0);
					endPoint.y = hitPoint(1);
					endPoint.z = hitPoint(2);
					line.points.push_back(startPoint);
					line.points.push_back(endPoint);

					line.scale.x = 0.03;
					line.scale.y = 0.03;
					line.scale.z = 0.03;
					line.color.a = 1.0;
					line.color.g = 1.0;
					line.lifetime = rclcpp::Duration::from_seconds(0.5);
					rayVisMsg.markers.push_back(line);
					++id;			
				}
			}
		}
		this->raycastVisPub_->publish(rayVisMsg);
		return true;
	}

	void occMap::depthPoseCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose){
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

		// store current position and orientation (depth sensor)
		Eigen::Matrix4d depthSensorPoseMatrix;
		this->getSensorPose(pose, this->body2DepthSensor_, depthSensorPoseMatrix);

		this->depthSensorPosition_(0) = depthSensorPoseMatrix(0, 3);
		this->depthSensorPosition_(1) = depthSensorPoseMatrix(1, 3);
		this->depthSensorPosition_(2) = depthSensorPoseMatrix(2, 3);
		this->depthSensorOrientation_ = depthSensorPoseMatrix.block<3, 3>(0, 0);

		if (this->isInMap(this->position_)){
			this->occNeedUpdate_ = true;
			this->depthUpdate_ = true;
		}
		else{
			this->occNeedUpdate_ = false;
			this->depthUpdate_ = false;
		}	
	}

	void occMap::depthOdomCB(const sensor_msgs::msg::Image::ConstSharedPtr& img, const nav_msgs::msg::Odometry::ConstSharedPtr& odom){
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
		Eigen::Matrix4d depthSensorPoseMatrix;
		this->getSensorPose(odom, this->body2DepthSensor_, depthSensorPoseMatrix);

		this->depthSensorPosition_(0) = depthSensorPoseMatrix(0, 3);
		this->depthSensorPosition_(1) = depthSensorPoseMatrix(1, 3);
		this->depthSensorPosition_(2) = depthSensorPoseMatrix(2, 3);
		this->depthSensorOrientation_ = depthSensorPoseMatrix.block<3, 3>(0, 0);

		if (this->isInMap(this->position_)){
			this->occNeedUpdate_ = true;
			this->depthUpdate_ = true;
		}
		else{
			this->occNeedUpdate_ = false;
			this->depthUpdate_ = false;
		}
	}

	void occMap::pointcloudPoseCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud, const geometry_msgs::msg::PoseStamped::ConstSharedPtr&pose){
		// directly get the point cloud
		pcl::PCLPointCloud2 pclPC2;
		pcl_conversions::toPCL(*pointcloud, pclPC2); // convert ros pointcloud2 to pcl pointcloud2
		pcl::fromPCLPointCloud2(pclPC2, this->pointcloud_);

		// store current robot position and orientation
		this->position_(0) = pose->pose.position.x;
		this->position_(1) = pose->pose.position.y;
		this->position_(2) = pose->pose.position.z;
      	Eigen::Quaterniond robotQuat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
		this->orientation_ = robotQuat.toRotationMatrix();

		// store current position and orientation (camera)
		Eigen::Matrix4d pointcloudSensorPoseMatrix;
		this->getSensorPose(pose, this->body2PointcloudSensor_, pointcloudSensorPoseMatrix);

		this->pointcloudSensorPosition_(0) = pointcloudSensorPoseMatrix(0, 3);
		this->pointcloudSensorPosition_(1) = pointcloudSensorPoseMatrix(1, 3);
		this->pointcloudSensorPosition_(2) = pointcloudSensorPoseMatrix(2, 3);
		this->pointcloudSensorOrientation_ = pointcloudSensorPoseMatrix.block<3, 3>(0, 0);

		if (this->isInMap(this->position_)){
			this->occNeedUpdate_ = true;
			this->pointcloudUpdate_ = true;
		}
		else{
			this->occNeedUpdate_ = false;
			this->pointcloudUpdate_ = false;
		}
	}

	void occMap::pointcloudOdomCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud, const nav_msgs::msg::Odometry::ConstSharedPtr& odom){
		// directly get the point cloud
		pcl::PCLPointCloud2 pclPC2;
		pcl_conversions::toPCL(*pointcloud, pclPC2); // convert ros pointcloud2 to pcl pointcloud2
		pcl::fromPCLPointCloud2(pclPC2, this->pointcloud_);

		// store current robot position and orientation
		this->position_(0) = odom->pose.pose.position.x;
		this->position_(1) = odom->pose.pose.position.y;
		this->position_(2) = odom->pose.pose.position.z;
      	Eigen::Quaterniond robotQuat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
		this->orientation_ = robotQuat.toRotationMatrix();

		// store current position and orientation (camera)
		Eigen::Matrix4d pointcloudSensorPoseMatrix;
		this->getSensorPose(odom, this->body2PointcloudSensor_, pointcloudSensorPoseMatrix);

		this->pointcloudSensorPosition_(0) = pointcloudSensorPoseMatrix(0, 3);
		this->pointcloudSensorPosition_(1) = pointcloudSensorPoseMatrix(1, 3);
		this->pointcloudSensorPosition_(2) = pointcloudSensorPoseMatrix(2, 3);
		this->pointcloudSensorOrientation_ = pointcloudSensorPoseMatrix.block<3, 3>(0, 0);


		if (this->isInMap(this->position_)){
			this->occNeedUpdate_ = true;
			this->pointcloudUpdate_ = true;
		}
		else{
			this->occNeedUpdate_ = false;
			this->pointcloudUpdate_ = false;
		}
	}

	void occMap::updateOccupancyCB(){
		if (not this->occNeedUpdate_){
			return;
		}
		// cout << "update occupancy map" << endl;
		rclcpp::Time startTime, endTime;
		
		startTime = rclcpp::Clock().now();
		if (this->depthUpdate_){
			// project 3D points from depth map
			this->projectDepthImage();
		}

		if (this->pointcloudUpdate_){
			// directly get pointcloud
			this->getPointcloud();
		}
		// raycasting and update occupancy
		this->raycastUpdate();

		// clear local map
		if (this->cleanLocalMap_){
			this->cleanLocalMap();
		}
		// clean dynamic obstacles
		this->cleanDynamicObstacles();
		// inflate map
		// this->inflateLocalMap();
		endTime = rclcpp::Clock().now();
		if (this->verbose_){
			cout << this->hint_ << ": Occupancy update time: " << (endTime - startTime).seconds() << " s." << endl;
		}
		this->occNeedUpdate_ = false;
		this->mapNeedInflate_ = true;
	}

	void occMap::inflateMapCB(){
		// inflate local map:
		if (this->mapNeedInflate_){
			this->inflateLocalMap();
			this->mapNeedInflate_ = false;
			this->esdfNeedUpdate_ = true;
		}
	}

	void occMap::staticClusteringCB(){
		if (this->obclustering_ == NULL){
			this->obclustering_.reset(new obstacleClustering (0.2)); // cloud resolution of 0.2
		}
		std::vector<Eigen::Vector3d> currCloud;
		double groudHeight = 0.4;
		double verticalRange = 2.0;
		double horizontalRange = 5.0;
		double cloudRes = 0.2;
		
		Eigen::Vector3d pOrigin = this->position_;
		// find four vextex of the bounding boxes
		double xStart = floor((pOrigin(0) - horizontalRange)/cloudRes)*cloudRes;
		double xEnd = ceil((pOrigin(0) + horizontalRange)/cloudRes)*cloudRes;
		double yStart = floor((pOrigin(1) - horizontalRange)/cloudRes)*cloudRes;
		double yEnd = ceil((pOrigin(1) + horizontalRange)/cloudRes)*cloudRes;
		double zStart = groudHeight;
		double zEnd = ceil((pOrigin(2) + verticalRange)/cloudRes)*cloudRes;
		for (double ix=xStart; ix<=xEnd; ix+=cloudRes){
			for (double iy=yStart; iy<=yEnd; iy+=cloudRes){
				for (double iz=zStart; iz<=zEnd; iz+=cloudRes){
					Eigen::Vector3d p (ix, iy, iz);
					Eigen::Vector3d diff = p - pOrigin;
					diff(2) = 0.0;
					if (diff.norm() <= horizontalRange){
						if (this->isInMap(p) and this->isInflatedOccupied(p)){
							currCloud.push_back(p);
						}
					}
				}
			}
		}
		this->currCloud_ = currCloud;
		this->obclustering_->run(currCloud);
		this->refinedBBoxVertices_ = this->obclustering_->getRefinedBBoxes();
	}

	void occMap::visCB(){
		this->publishProjPoints();
		this->publishMap();
		this->publishInflatedMap();
		this->publish2DOccupancyGrid();
		this->publishStaticObstacles();
	}

	void occMap::projectDepthImage(){
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
				
				// if (*rowPtr == 0) {
				// 	depth = this->raycastMaxLength_ + 0.1;
				// 	rowPtr =  rowPtr + this->skipPixel_;
				// 	continue;
				// } else if (depth < this->depthMinValue_) {
				// 	continue;
				// } else if (depth > this->depthMaxValue_ and depth < 1.5 * this->depthMaxValue_) {
				// 	depth = this->raycastMaxLength_ + 0.1;
				// }
				// else if (depth >= 1.5 * this->depthMaxValue_){
				// 	rowPtr =  rowPtr + this->skipPixel_;
				// 	continue;
				// }

				if (*rowPtr == 0) {
					depth = this->raycastMaxLength_ + 0.1;
				} else if (depth < this->depthMinValue_) {
					continue;
				} else if (depth > this->depthMaxValue_ ) {
					depth = this->raycastMaxLength_ + 0.1;
				}

				rowPtr =  rowPtr + this->skipPixel_;

				// get 3D point in camera frame
				currPointCam(0) = (u - this->cx_) * depth * inv_fx;
				currPointCam(1) = (v - this->cy_) * depth * inv_fy;
				currPointCam(2) = depth;
				currPointMap = this->depthSensorOrientation_ * currPointCam + this->depthSensorPosition_; // transform to map coordinate

				if (this->useFreeRegions_){ // this region will not be updated and directly set to free
					if (this->isInHistFreeRegions(currPointMap)){
						continue;
					}
				}

				// store current point
				this->projPoints_[this->projPointsNum_] = currPointMap;
				this->projPointsNum_ = this->projPointsNum_ + 1;
			}
		} 
	}

	void occMap::getPointcloud(){
		this->pointcloudSensorPointsNum_ = 0;
		this->pointcloudSensorPoints_.resize(this->pointcloud_.size());
		Eigen::Vector3d currPointSensor, currPointMap;
		for (int i=0; i<int(this->pointcloud_.size()); ++i){
			currPointSensor(0) = this->pointcloud_.points[i].x;
			currPointSensor(1) = this->pointcloud_.points[i].y;
			currPointSensor(2) = this->pointcloud_.points[i].z;
			currPointMap = this->pointcloudSensorOrientation_ * currPointSensor + this->pointcloudSensorPosition_; // transform to map coordinate
			double dist = (currPointMap-this->pointcloudSensorPosition_).norm();
			if (dist >= this->pointcloudMinDistance_ and dist <= this->pointcloudMaxDistance_){
				this->pointcloudSensorPoints_[i] = currPointMap;
				++this->pointcloudSensorPointsNum_;
			}
		}
	}

	void occMap::raycastUpdate(){
		if (this->projPointsNum_ == 0 and this->pointcloudSensorPointsNum_ == 0){
			return;
		}
		this->raycastNum_ += 1;

		// record local bound of update
		double xmin, xmax, ymin, ymax, zmin, zmax;
		xmin = xmax = this->position_(0);
		ymin = ymax = this->position_(1);
		zmin = zmax = this->position_(2);

		// iterate through each projected points, perform raycasting and update occupancy
		Eigen::Vector3d currPoint;
		bool pointAdjusted;
		int rayendVoxelID, raycastVoxelID;
		double length;
		for (int i=0; i<this->projPointsNum_+this->pointcloudSensorPointsNum_; ++i){
			if (i < projPointsNum_){
				currPoint = this->projPoints_[i];
			}
			else{
				currPoint = this->pointcloudSensorPoints_[i - this->projPointsNum_];
			}

			if (std::isnan(currPoint(0)) or std::isnan(currPoint(1)) or std::isnan(currPoint(2))){
				continue; // nan points can happen when we are using pointcloud as input
			}

			pointAdjusted = false;
			// check whether the point is in reserved map range
			if (not this->isInMap(currPoint)){
				currPoint = this->adjustPointInMap(currPoint);
				pointAdjusted = true;
			}

			// check whether the point exceeds the maximum raycasting length
			length = (currPoint - this->position_).norm();
			if (length > this->raycastMaxLength_){
				currPoint = this->adjustPointRayLength(currPoint);
				pointAdjusted = true;
			}

			// update local bound
			if (currPoint(0) < xmin){xmin = currPoint(0);}
			if (currPoint(1) < ymin){ymin = currPoint(1);}
			if (currPoint(2) < zmin){zmin = currPoint(2);}
			if (currPoint(0) > xmax){xmax = currPoint(0);}
			if (currPoint(1) > ymax){ymax = currPoint(1);}
			if (currPoint(2) > zmax){zmax = currPoint(2);}

			// update occupancy itself update information
			rayendVoxelID = this->updateOccupancyInfo(currPoint, not pointAdjusted); // point adjusted is free, not is occupied

			// check whether the voxel has already been updated, so no raycasting needed
			// rayendVoxelID = this->posToAddress(currPoint);
			if (this->flagRayend_[rayendVoxelID] == this->raycastNum_){
				continue; // skip
			}
			else{
				this->flagRayend_[rayendVoxelID] = this->raycastNum_;
			}

			// raycasting for update occupancy
			this->raycaster_.setInput(currPoint/this->mapRes_, this->position_/this->mapRes_);
			Eigen::Vector3d rayPoint, actualPoint;
			while (this->raycaster_.step(rayPoint)){
				actualPoint = rayPoint;
				actualPoint(0) += 0.5;
				actualPoint(1) += 0.5;
				actualPoint(2) += 0.5;
				actualPoint *= this->mapRes_;
				raycastVoxelID = this->updateOccupancyInfo(actualPoint, false);

				// raycastVoxelID = this->posToAddress(actualPoint);
				if (this->flagTraverse_[raycastVoxelID] == this->raycastNum_){
					break;
				}
				else{
					this->flagTraverse_[raycastVoxelID] = this->raycastNum_;
				}

			}
		}

		// store local bound and inflate local bound (inflate is for ESDF update)
		this->posToIndex(Eigen::Vector3d (xmin, ymin, zmin), this->localBoundMin_);
		this->posToIndex(Eigen::Vector3d (xmax, ymax, zmax), this->localBoundMax_);
		this->localBoundMin_ -= int(ceil(this->localBoundInflate_/this->mapRes_)) * Eigen::Vector3i(1, 1, 0); // inflate in x y direction
		this->localBoundMax_ += int(ceil(this->localBoundInflate_/this->mapRes_)) * Eigen::Vector3i(1, 1, 0); 
		this->boundIndex(this->localBoundMin_); // since inflated, need to bound if not in reserved range
		this->boundIndex(this->localBoundMax_);


		// update occupancy in the cache
		double logUpdateValue;
		int cacheAddress, hit, miss;
		while (not this->updateVoxelCache_.empty()){
			Eigen::Vector3i cacheIdx = this->updateVoxelCache_.front();
			this->updateVoxelCache_.pop();
			cacheAddress = this->indexToAddress(cacheIdx);

			hit = this->countHit_[cacheAddress];
			miss = this->countHitMiss_[cacheAddress] - hit;

			if (hit >= miss and hit != 0){
				logUpdateValue = this->pHitLog_;
			}
			else{
				logUpdateValue = this->pMissLog_;
			}
			this->countHit_[cacheAddress] = 0; // clear hit
			this->countHitMiss_[cacheAddress] = 0; // clear hit and miss

			// check whether point is in the local update range
			if (not this->isInLocalUpdateRange(cacheIdx)){
				continue; // do not update if not in the range
			}

			if (this->useFreeRegions_){ // current used in simulation, this region will not be updated and directly set to free
				Eigen::Vector3d pos;
				this->indexToPos(cacheIdx, pos);
				if (this->isInHistFreeRegions(pos)){
					this->occupancy_[cacheAddress] = this->pMinLog_;
					continue;
				}
			}

			// update occupancy info
			if ((logUpdateValue >= 0) and (this->occupancy_[cacheAddress] >= this->pMaxLog_)){
				continue; // not increase p if max clamped
			}
			else if ((logUpdateValue <= 0) and (this->occupancy_[cacheAddress] == this->pMinLog_)){
				continue; // not decrease p if min clamped
			}
			else if ((logUpdateValue <= 0) and (this->occupancy_[cacheAddress] < this->pMinLog_)){
				this->occupancy_[cacheAddress] = this->pMinLog_; // if unknown set it free (prior), 
				continue;
			}

			this->occupancy_[cacheAddress] = std::min(std::max(this->occupancy_[cacheAddress]+logUpdateValue, this->pMinLog_), this->pMaxLog_);

			// update the entire map range (if it is not unknown)
			if (not this->isUnknown(cacheIdx)){
				Eigen::Vector3d cachePos;
				this->indexToPos(cacheIdx, cachePos);
				if (cachePos(0) > this->currMapRangeMax_(0)){
					this->currMapRangeMax_(0) = cachePos(0);
				}
				else if (cachePos(0) < this->currMapRangeMin_(0)){
					this->currMapRangeMin_(0) = cachePos(0);
				}

				if (cachePos(1) > this->currMapRangeMax_(1)){
					this->currMapRangeMax_(1) = cachePos(1);
				}
				else if (cachePos(1) < this->currMapRangeMin_(1)){
					this->currMapRangeMin_(1) = cachePos(1);
				}

				if (cachePos(2) > this->currMapRangeMax_(2)){
					this->currMapRangeMax_(2) = cachePos(2);
				}
				else if (cachePos(2) < this->currMapRangeMin_(2)){
					this->currMapRangeMin_(2) = cachePos(2);
				}
			}
		}
	}   

	void occMap::cleanLocalMap(){
		Eigen::Vector3i posIndex;
		this->posToIndex(this->position_, posIndex);
		Eigen::Vector3i innerMinBBX = posIndex - this->localMapVoxel_;
		Eigen::Vector3i innerMaxBBX = posIndex + this->localMapVoxel_;
		Eigen::Vector3i outerMinBBX = innerMinBBX - Eigen::Vector3i(5, 5, 5);
		Eigen::Vector3i outerMaxBBX = innerMaxBBX + Eigen::Vector3i(5, 5, 5);
		this->boundIndex(innerMinBBX);
		this->boundIndex(innerMaxBBX);
		this->boundIndex(outerMinBBX);
		this->boundIndex(outerMaxBBX);

		// clear x axis
		for (int y=outerMinBBX(1); y<=outerMaxBBX(1); ++y){
			for (int z=outerMinBBX(2); z<=outerMaxBBX(2); ++z){
				for (int x=outerMinBBX(0); x<=innerMinBBX(0); ++x){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;
				}

				for (int x=innerMaxBBX(0); x<=outerMaxBBX(0); ++x){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;					
				}
			}
		}

		// clear y axis
		for (int x=outerMinBBX(0); x<=outerMaxBBX(0); ++x){
			for (int z=outerMinBBX(2); z<=outerMaxBBX(2); ++z){
				for (int y=outerMinBBX(1); y<=innerMinBBX(1); ++y){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;
				}

				for (int y=innerMaxBBX(1); y<=outerMaxBBX(1); ++y){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;
				}
			}
		}

		// clear z axis
		for (int x=outerMinBBX(0); x<=outerMaxBBX(0); ++x){
			for (int y=outerMinBBX(1); y<=outerMaxBBX(1); ++y){
				for (int z=outerMinBBX(2); z<=innerMinBBX(2); ++z){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;
				}

				for (int z=innerMaxBBX(2); z<=outerMaxBBX(2); ++z){
					this->occupancy_[this->indexToAddress(Eigen::Vector3i (x, y, z))] = this->pMinLog_ - this->UNKNOWN_FLAG_;
				}
			}
		}
	}

	void occMap::inflateLocalMap(){
		int xmin = this->localBoundMin_(0);
		int xmax = this->localBoundMax_(0);
		int ymin = this->localBoundMin_(1);
		int ymax = this->localBoundMax_(1);
		int zmin = this->localBoundMin_(2);
		int zmax = this->localBoundMax_(2);
		Eigen::Vector3i clearIndex;
		// clear previous data in current data range
		for (int x=xmin; x<=xmax; ++x){
			for (int y=ymin; y<=ymax; ++y){
				for (int z=zmin; z<=zmax; ++z){
					clearIndex(0) = x; clearIndex(1) = y; clearIndex(2) = z;
					this->occupancyInflated_[this->indexToAddress(clearIndex)] = false;
				}
			}
		}

		int xInflateSize = ceil(this->robotSize_(0)/(2*this->mapRes_));
		int yInflateSize = ceil(this->robotSize_(1)/(2*this->mapRes_));
		int zInflateSize = ceil(this->robotSize_(2)/(2*this->mapRes_));

		// inflate based on current occupancy
		Eigen::Vector3i pointIndex, inflateIndex;
		int inflateAddress;
		const int  maxIndex = this->mapVoxelMax_(0) * this->mapVoxelMax_(1) * this->mapVoxelMax_(2);
		for (int x=xmin; x<=xmax; ++x){
			for (int y=ymin; y<=ymax; ++y){
				for (int z=zmin; z<=zmax; ++z){
					pointIndex(0) = x; pointIndex(1) = y; pointIndex(2) = z;
					if (this->isOccupied(pointIndex)){
						for (int ix=-xInflateSize; ix<=xInflateSize; ++ix){
							for (int iy=-yInflateSize; iy<=yInflateSize; ++iy){
								for (int iz=-zInflateSize; iz<=zInflateSize; ++iz){
									inflateIndex(0) = pointIndex(0) + ix;
									inflateIndex(1) = pointIndex(1) + iy;
									inflateIndex(2) = pointIndex(2) + iz;
									inflateAddress = this->indexToAddress(inflateIndex);
									if ((inflateAddress < 0) or (inflateAddress > maxIndex)){
										continue; // those points are not in the reserved map
									} 
									this->occupancyInflated_[inflateAddress] = true;
								}
							}
						}
					}
				}
			}
		}
	}

    void occMap::cleanDynamicObstacles(){
		if (this->getDynamicObstaclesClient_->service_is_ready()){
        	auto request = std::make_shared<onboard_detector::srv::GetDynamicObstacles::Request>();
			request->current_position.x = this->position_(0);
			request->current_position.y = this->position_(1);
			request->current_position.z = this->position_(2);
			request->range = this->raycastMaxLength_;	


			auto result = this->getDynamicObstaclesClient_->async_send_request(request);
			// if (rclcpp::spin_until_future_complete(this->getDynamicObstacleSrvNode_, result) == rclcpp::FutureReturnCode::SUCCESS){
        	std::future_status status = result.wait_for(0.001s);  // timeout to guarantee a graceful finish
			while (status != std::future_status::ready){
				status = result.wait_for(0.001s);
			}
			if (status == std::future_status::ready){
				auto response = result.get();
				
				std::vector<geometry_msgs::msg::Vector3> obsPos = response->position;
				std::vector<geometry_msgs::msg::Vector3> obsVel = response->velocity;
				std::vector<geometry_msgs::msg::Vector3> obsSize = response->size;

				std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> freeRegions;
				for (size_t i = 0; i<obsPos.size(); ++i) {
					Eigen::Vector3d lowerBound(obsPos[i].x - obsSize[i].x / 2 - 0.3,
											obsPos[i].y - obsSize[i].y / 2 - 0.3,
											std::max(0.0, obsPos[i].z - obsSize[i].z / 2 - 0.3));
					Eigen::Vector3d upperBound(obsPos[i].x + obsSize[i].x / 2 + 0.3,
											obsPos[i].y + obsSize[i].y / 2 + 0.3,
											obsPos[i].z + obsSize[i].z / 2 + 0.3);
					freeRegions.push_back(std::make_pair(lowerBound, upperBound));
				}
				this->freeRegions(freeRegions);
				this->updateFreeRegions(freeRegions);
			}
		}
	}


	void occMap::publishProjPoints(){
		pcl::PointXYZ pt;
		pcl::PointCloud<pcl::PointXYZ> cloud;

		for (int i=0; i<this->projPointsNum_; ++i){
			pt.x = this->projPoints_[i](0);
			pt.y = this->projPoints_[i](1);
			pt.z = this->projPoints_[i](2);
			cloud.push_back(pt);
		}

		cloud.width = cloud.points.size();
		cloud.height = 1;
		cloud.is_dense = true;
		cloud.header.frame_id = "map";

		sensor_msgs::msg::PointCloud2 cloudMsg;
		pcl::toROSMsg(cloud, cloudMsg);
		this->depthCloudPub_->publish(cloudMsg);
	}

	void occMap::publishMap(){
		pcl::PointXYZ pt;
		pcl::PointCloud<pcl::PointXYZ> cloud;

		Eigen::Vector3d minRange, maxRange;
		if (this->visGlobalMap_){
			// minRange = this->mapSizeMin_;
			// maxRange = this->mapSizeMax_;
			minRange = this->currMapRangeMin_;
			maxRange = this->currMapRangeMax_;
		}
		else{
			minRange = this->position_ - localMapSize_;
			maxRange = this->position_ + localMapSize_;
			minRange(2) = this->groundHeight_;
		}
		Eigen::Vector3i minRangeIdx, maxRangeIdx;
		this->posToIndex(minRange, minRangeIdx);
		this->posToIndex(maxRange, maxRangeIdx);
		this->boundIndex(minRangeIdx);
		this->boundIndex(maxRangeIdx);

		for (int x=minRangeIdx(0); x<=maxRangeIdx(0); ++x){
			for (int y=minRangeIdx(1); y<=maxRangeIdx(1); ++y){
				for (int z=minRangeIdx(2); z<=maxRangeIdx(2); ++z){
					Eigen::Vector3i pointIdx (x, y, z);

					// if (this->occupancy_[this->indexToAddress(pointIdx)] > this->pMinLog_){
					if (this->isOccupied(pointIdx)){
						Eigen::Vector3d point;
						this->indexToPos(pointIdx, point);
						if (point(2) <= this->maxVisHeight_){
							pt.x = point(0);
							pt.y = point(1);
							pt.z = point(2);
							cloud.push_back(pt);
						}
					}
				}
			}
		}

		cloud.width = cloud.points.size();
		cloud.height = 1;
		cloud.is_dense = true;
		cloud.header.frame_id = "map";

		sensor_msgs::msg::PointCloud2 cloudMsg;
		pcl::toROSMsg(cloud, cloudMsg);
		this->mapVisPub_->publish(cloudMsg);
	}

	void occMap::publishInflatedMap(){
		pcl::PointXYZ pt;
		pcl::PointCloud<pcl::PointXYZ> cloud;

		Eigen::Vector3d minRange, maxRange;
		if (this->visGlobalMap_){
			// minRange = this->mapSizeMin_;
			// maxRange = this->mapSizeMax_;
			minRange = this->currMapRangeMin_;
			maxRange = this->currMapRangeMax_;
		}
		else{
			minRange = this->position_ - localMapSize_;
			maxRange = this->position_ + localMapSize_;
			minRange(2) = this->groundHeight_;
		}
		Eigen::Vector3i minRangeIdx, maxRangeIdx;
		this->posToIndex(minRange, minRangeIdx);
		this->posToIndex(maxRange, maxRangeIdx);
		this->boundIndex(minRangeIdx);
		this->boundIndex(maxRangeIdx);

		for (int x=minRangeIdx(0); x<=maxRangeIdx(0); ++x){
			for (int y=minRangeIdx(1); y<=maxRangeIdx(1); ++y){
				for (int z=minRangeIdx(2); z<=maxRangeIdx(2); ++z){
					Eigen::Vector3i pointIdx (x, y, z);
					// if (this->occupancy_[this->indexToAddress(pointIdx)] > this->pMinLog_){
					if (this->isInflatedOccupied(pointIdx)){
						Eigen::Vector3d point;
						this->indexToPos(pointIdx, point);
						if (point(2) <= this->maxVisHeight_){
							pt.x = point(0);
							pt.y = point(1);
							pt.z = point(2);
							cloud.push_back(pt);
						}
					}
				}
			}
		}

		cloud.width = cloud.points.size();
		cloud.height = 1;
		cloud.is_dense = true;
		cloud.header.frame_id = "map";

		sensor_msgs::msg::PointCloud2 cloudMsg;
		pcl::toROSMsg(cloud, cloudMsg);
		this->inflatedMapVisPub_->publish(cloudMsg);	
	}

	void occMap::publish2DOccupancyGrid(){
		Eigen::Vector3d minRange, maxRange;
		minRange = this->mapSizeMin_;
		maxRange = this->mapSizeMax_;
		minRange(2) = this->groundHeight_;
		Eigen::Vector3i minRangeIdx, maxRangeIdx;
		this->posToIndex(minRange, minRangeIdx);
		this->posToIndex(maxRange, maxRangeIdx);
		this->boundIndex(minRangeIdx);
		this->boundIndex(maxRangeIdx);

		nav_msgs::msg::OccupancyGrid mapMsg;
		for (int i=0; i<maxRangeIdx(0); ++i){
			for (int j=0; j<maxRangeIdx(1); ++j){
				mapMsg.data.push_back(0);
			}
		}

		double z = 0.5;
		int zIdx = int(z/this->mapRes_);
		for (int x=minRangeIdx(0); x<=maxRangeIdx(0); ++x){
			for (int y=minRangeIdx(1); y<=maxRangeIdx(1); ++y){
				Eigen::Vector3i pointIdx (x, y, zIdx);
				int map2DIdx = x  +  y * maxRangeIdx(0);
				if (this->isUnknown(pointIdx)){
					mapMsg.data[map2DIdx] = -1;
				}
				else if (this->isOccupied(pointIdx)){
					mapMsg.data[map2DIdx] = 100;
				}
				else{
					mapMsg.data[map2DIdx] = 0;
				}
			}
		}
		mapMsg.header.frame_id = "map";
		mapMsg.header.stamp = this->get_clock()->now();
		mapMsg.info.resolution = this->mapRes_;
		mapMsg.info.width = maxRangeIdx(0);
		mapMsg.info.height = maxRangeIdx(1);
		mapMsg.info.origin.position.x = minRange(0);
		mapMsg.info.origin.position.y = minRange(1);
		this->map2DPub_->publish(mapMsg);		
	}

	void occMap::publishStaticObstacles(){
		if (this->refinedBBoxVertices_.size() != 0){
		    visualization_msgs::msg::Marker line;
		    visualization_msgs::msg::MarkerArray lines;

		    line.header.frame_id = "map";
		    line.type = visualization_msgs::msg::Marker::LINE_LIST;
		    line.action = visualization_msgs::msg::Marker::ADD;
		    line.ns = "map_static_obstacles";  
		    line.scale.x = 0.06;
		    line.color.r = 0;
		    line.color.g = 1;
		    line.color.b = 1;
		    line.color.a = 1.0;
		    line.lifetime = rclcpp::Duration::from_seconds(0.2);
		    Eigen::Vector3d vertex_pose;
		    for(int i=0; i<int(this->refinedBBoxVertices_.size()); ++i){
		        bboxVertex v = this->refinedBBoxVertices_[i];
		        std::vector<geometry_msgs::msg::Point> verts;
		        geometry_msgs::msg::Point p;

				for (int j=0; j<int(v.vert.size());++j){
					p.x = v.vert[j](0); p.y = v.vert[j](1); p.z = v.vert[j](2);
		        	verts.push_back(p);
				}

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
		        for (int j=0;j<12;++j){
		            line.points.push_back(verts[vert_idx[j][0]]);
		            line.points.push_back(verts[vert_idx[j][1]]);
		        }
		        lines.markers.push_back(line);
		        line.id++;
		    }
		    this->staticObstacleVisPub_->publish(lines);		
		}
	}
}