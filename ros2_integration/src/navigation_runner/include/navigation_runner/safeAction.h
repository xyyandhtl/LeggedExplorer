/*
	FILE: safeAction.cpp
	--------------------------------------
	safe action header file
*/
#ifndef SAFE_ACTION_H
#define SAFE_ACTION_H
#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Eigen>
#include <navigation_runner/solver.h>
#include <navigation_runner/utils.h>
#include <navigation_runner/srv/get_safe_action.hpp>
#include <onboard_detector/detectors/dbscan.h>
#include <visualization_msgs/msg/marker_array.hpp>

using std::cout; using std::endl;
namespace navigationRunner{
    class safeAction : public rclcpp::Node{
    private:
		std::string ns_;
		std::string hint_;

        // ROS
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr staticObsVisPub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dynObsVisPub_;
        rclcpp::TimerBase::SharedPtr staticObsVisTimer_;
        rclcpp::TimerBase::SharedPtr dynObsVisTimer_;
        rclcpp::Service<navigation_runner::srv::GetSafeAction>::SharedPtr getSafeActionServer_;

        // clustering
        std::shared_ptr<onboardDetector::DBSCAN> dbCluster_;

        // parameters
		double timeHorizon_;
		double timeStep_;
		double minHeight_;
		double maxHeight_;
		double safetyDist_;
		bool useSafetyInStatic_;

        // data
		std::vector<Vector3> staticObsPosVec_;
		std::vector<double> staticObsSizeVec_;
		std::vector<Vector3> dynObsPosVec_;
		std::vector<double> dynObsSizeVec_;

    public:
        safeAction();
        void initParam();
		Plane getORCAPlane(const Vector3& agentPos, const Vector3& agentVel, double agentRadius,
						   const Vector3& obsPos, const Vector3& obsVel, double obsRadius, bool useCircle, bool& inVO);    
		bool getSafeAction(const std::shared_ptr<navigation_runner::srv::GetSafeAction::Request> req, 
						   const std::shared_ptr<navigation_runner::srv::GetSafeAction::Response> res);
        void getClusters(const std::vector<Eigen::Vector3d>& points, std::vector<std::vector<Eigen::Vector3d>>& clusters);
        void staticObsVisCallback();
        void dynObsVisCallback();
    };
}

#endif