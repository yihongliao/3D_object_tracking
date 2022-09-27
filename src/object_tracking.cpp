#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>


#include <mutex>
#include <thread>

// PCL specific includes
#include <pcl/common/centroid.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/tracking.h>
#include <pcl/visualization/pcl_visualizer.h>

#define FPS_CALC_BEGIN                                                                 \
  static double duration = 0;                                                          \
  double start_time = pcl::getTime();

// clang-format off
#define FPS_CALC_END(_WHAT_)                                                           \
  {                                                                                    \
    double end_time = pcl::getTime();                                                  \
    static unsigned count = 0;                                                         \
    if (++count == 10) {                                                               \
      std::cout << "Average framerate(" << _WHAT_ << "): "                             \
                << double(count) / double(duration) << " Hz" << std::endl;             \
      count = 0;                                                                       \
      duration = 0.0;                                                                  \
    }                                                                                  \
    else {                                                                             \
      duration += end_time - start_time;                                               \
    }                                                                                  \
  }

using namespace pcl::tracking;
using namespace std::chrono_literals;

template <typename PointType> class SegmentTracking 
{
// ROS
public:
    ros::NodeHandle nh; 
    ros::Publisher pub;
    ros::Publisher pub2;
    ros::Publisher marker_pub;
    ros::Subscriber sub;
    tf2_ros::TransformBroadcaster tfb;

// PCL
public:
    using RefPointType = pcl::PointXYZRGB;
    using ParticleT = ParticleXYZRPY;

    using Cloud = pcl::PointCloud<PointType>;
    using RefCloud = pcl::PointCloud<RefPointType>;
    using RefCloudPtr = RefCloud::Ptr;
    using RefCloudConstPtr = RefCloud::ConstPtr;
    using CloudPtr = typename Cloud::Ptr;
    using CloudConstPtr = typename Cloud::ConstPtr;
    using ParticleFilter = ParticleFilterTracker<RefPointType, ParticleT>;
    using CoherencePtr = ParticleFilter::CoherencePtr;
    using KdTree = pcl::search::KdTree<PointType>;
    using KdTreePtr = typename KdTree::Ptr;

    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    CloudPtr cloud_pass_;
    CloudPtr cloud_pass_downsampled_;
    CloudPtr plane_cloud_;
    CloudPtr nonplane_cloud_;
    CloudPtr cloud_hull_;
    CloudPtr segmented_cloud_;
    CloudPtr reference_;
    std::vector<pcl::Vertices> hull_vertices_;

    std::mutex mtx_;
    bool new_cloud_;
    pcl::NormalEstimationOMP<PointType, pcl::Normal> ne_; // to store threadpool
    ParticleFilter::Ptr tracker_;
    int counter_;
    bool use_convex_hull_;
    bool visualize_non_downsample_;
    bool visualize_particles_;
    double tracking_time_;
    double computation_time_;
    double downsampling_time_;
    double downsampling_grid_size_;

    SegmentTracking(
                    int thread_nr,
                    double downsampling_grid_size,
                    bool use_convex_hull,
                    bool visualize_non_downsample,
                    bool visualize_particles,
                    bool use_fixed)
    : new_cloud_(false)
    , ne_(thread_nr)
    , counter_(0)
    , use_convex_hull_(use_convex_hull)
    , visualize_non_downsample_(visualize_non_downsample)
    , visualize_particles_(visualize_particles)
    , downsampling_grid_size_(downsampling_grid_size)
    {
        //ROS
        // Topic you want to publish
        marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
        pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
        pub2 = nh.advertise<sensor_msgs::PointCloud2> ("output_filtered", 1);

        // Topic you want to subscribe
        sub = nh.subscribe ("point_cloud", 1, &SegmentTracking<PointType>::cloud_cb, this);     

        //PCl
        KdTreePtr tree(new KdTree(false));
        ne_.setSearchMethod(tree);
        ne_.setRadiusSearch(0.03);

        std::vector<double> default_step_covariance = std::vector<double>(6, 0.015 * 0.015);
        default_step_covariance[3] *= 40.0;
        default_step_covariance[4] *= 40.0;
        default_step_covariance[5] *= 40.0;

        std::vector<double> initial_noise_covariance = std::vector<double>(6, 0.00001);
        std::vector<double> default_initial_mean = std::vector<double>(6, 0.0);
        if (use_fixed) {
            ParticleFilterOMPTracker<RefPointType, ParticleT>::Ptr tracker(
                new ParticleFilterOMPTracker<RefPointType, ParticleT>(thread_nr));
            tracker_ = tracker;
        }
        else {
            KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT>::Ptr tracker(
                new KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT>(thread_nr));
            tracker->setMaximumParticleNum(500);
            tracker->setDelta(0.99);
            tracker->setEpsilon(0.2);
            ParticleT bin_size;
            bin_size.x = 0.1f;
            bin_size.y = 0.1f;
            bin_size.z = 0.1f;
            bin_size.roll = 0.1f;
            bin_size.pitch = 0.1f;
            bin_size.yaw = 0.1f;
            tracker->setBinSize(bin_size);
            tracker_ = tracker;
        }

        tracker_->setTrans(Eigen::Affine3f::Identity());
        tracker_->setStepNoiseCovariance(default_step_covariance);
        tracker_->setInitialNoiseCovariance(initial_noise_covariance);
        tracker_->setInitialNoiseMean(default_initial_mean);
        tracker_->setIterationNum(1);

        tracker_->setParticleNum(400);
        tracker_->setResampleLikelihoodThr(0.00);
        tracker_->setUseNormal(false);
        // setup coherences
        ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence =
            ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr(
                new ApproxNearestPairPointCloudCoherence<RefPointType>());

        DistanceCoherence<RefPointType>::Ptr distance_coherence(
            new DistanceCoherence<RefPointType>);
        coherence->addPointCoherence(distance_coherence);

        HSVColorCoherence<RefPointType>::Ptr color_coherence(
            new HSVColorCoherence<RefPointType>);
        color_coherence->setWeight(0.1);
        coherence->addPointCoherence(color_coherence);

        pcl::search::Octree<RefPointType>::Ptr search(
            new pcl::search::Octree<RefPointType>(0.01));
        coherence->setSearchMethod(search);
        coherence->setMaximumDistance(0.01);
        tracker_->setCloudCoherence(coherence);
    }   

    void filterPassThrough(const CloudConstPtr& cloud, Cloud& result) 
    {
        FPS_CALC_BEGIN;
        pcl::PassThrough<PointType> pass;
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, 10.0);
        pass.setKeepOrganized(false);
        pass.setInputCloud(cloud);
        pass.filter(result);
        FPS_CALC_END("filterPassThrough");
    }

    void euclideanSegment(const CloudConstPtr& cloud,
                std::vector<pcl::PointIndices>& cluster_indices)
    {
        FPS_CALC_BEGIN;
        pcl::EuclideanClusterExtraction<PointType> ec;
        KdTreePtr tree(new KdTree());
        ec.setClusterTolerance(0.05); // 5cm
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        FPS_CALC_END("euclideanSegmentation");
    }

    void gridSample(const CloudConstPtr& cloud, Cloud& result, double leaf_size = 0.01)
    {
        FPS_CALC_BEGIN;
        double start = pcl::getTime();
        pcl::VoxelGrid<PointType> grid;
        grid.setLeafSize(float(leaf_size), float(leaf_size), float(leaf_size));
        grid.setInputCloud(cloud);
        grid.filter(result);
        double end = pcl::getTime();
        downsampling_time_ = end - start;
        FPS_CALC_END("gridSample");
    }

    void gridSampleApprox(const CloudConstPtr& cloud, Cloud& result, double leaf_size = 0.01)
    {
        FPS_CALC_BEGIN;
        double start = pcl::getTime();
        pcl::ApproximateVoxelGrid<PointType> grid;
        grid.setLeafSize(static_cast<float>(leaf_size),
                            static_cast<float>(leaf_size),
                            static_cast<float>(leaf_size));
        grid.setInputCloud(cloud);
        grid.filter(result);
        double end = pcl::getTime();
        downsampling_time_ = end - start;
        FPS_CALC_END("gridSample");
    }

    void planeSegmentation(const CloudConstPtr& cloud,
                        pcl::ModelCoefficients& coefficients,
                        pcl::PointIndices& inliers)
    {
        FPS_CALC_BEGIN;
        pcl::SACSegmentation<PointType> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.005);
        seg.setInputCloud(cloud);
        seg.segment(inliers, coefficients);
        FPS_CALC_END("planeSegmentation");
    }

    void planeProjection(const CloudConstPtr& cloud,
                    Cloud& result,
                    const pcl::ModelCoefficients::ConstPtr& coefficients)
    {
        FPS_CALC_BEGIN;
        pcl::ProjectInliers<PointType> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(cloud);
        proj.setModelCoefficients(coefficients);
        proj.filter(result);
        FPS_CALC_END("planeProjection");
    }

    void convexHull(const CloudConstPtr& cloud,
                Cloud&,
                std::vector<pcl::Vertices>& hull_vertices)
    {
        FPS_CALC_BEGIN;
        pcl::ConvexHull<PointType> chull;
        chull.setInputCloud(cloud);
        chull.reconstruct(*cloud_hull_, hull_vertices);
        FPS_CALC_END("convexHull");
    }

    void normalEstimation(const CloudConstPtr& cloud, pcl::PointCloud<pcl::Normal>& result)
    {
        FPS_CALC_BEGIN;
        ne_.setInputCloud(cloud);
        ne_.compute(result);
        FPS_CALC_END("normalEstimation");
    }

    void tracking(const RefCloudConstPtr& cloud)
    {
        double start = pcl::getTime();
        FPS_CALC_BEGIN;
        tracker_->setInputCloud(cloud);
        tracker_->compute();
        double end = pcl::getTime();
        FPS_CALC_END("tracking");
        tracking_time_ = end - start;
    }

    void addNormalToCloud(const CloudConstPtr& cloud,
                    const pcl::PointCloud<pcl::Normal>::ConstPtr&,
                    RefCloud& result)
    {
        result.width = cloud->width;
        result.height = cloud->height;
        result.is_dense = cloud->is_dense;
        for (const auto& pt : *cloud) {
        RefPointType point;
        point.x = pt.x;
        point.y = pt.y;
        point.z = pt.z;
        point.rgba = pt.rgba;
        result.push_back(point);
        }
    }

    void extractNonPlanePoints(const CloudConstPtr& cloud,
                            const CloudConstPtr& cloud_hull,
                            Cloud& result)
    {
        pcl::ExtractPolygonalPrismData<PointType> polygon_extract;
        pcl::PointIndices::Ptr inliers_polygon(new pcl::PointIndices());
        polygon_extract.setHeightLimits(0.01, 10.0);
        polygon_extract.setInputPlanarHull(cloud_hull);
        polygon_extract.setInputCloud(cloud);
        polygon_extract.segment(*inliers_polygon);
        {
        pcl::ExtractIndices<PointType> extract_positive;
        extract_positive.setNegative(false);
        extract_positive.setInputCloud(cloud);
        extract_positive.setIndices(inliers_polygon);
        extract_positive.filter(result);
        }
    }

    void removeZeroPoints(const CloudConstPtr& cloud, Cloud& result)
    {
        for (const auto& point : *cloud) {
        if (!(std::abs(point.x) < 0.01 && std::abs(point.y) < 0.01 &&
                std::abs(point.z) < 0.01) &&
            !std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z))
            result.push_back(point);
        }

        result.width = result.size();
        result.height = 1;
        result.is_dense = true;
    }

    void extractSegmentCluster(const CloudConstPtr& cloud,
                            const std::vector<pcl::PointIndices>& cluster_indices,
                            const int segment_index,
                            Cloud& result)
    {
        pcl::PointIndices segmented_indices = cluster_indices[segment_index];
        for (const auto& index : segmented_indices.indices) {
        PointType point = (*cloud)[index];
        result.push_back(point);
        }
        result.width = result.size();
        result.height = 1;
        result.is_dense = true;
    }

    void drawParticles(pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud)
    {
        ParticleFilter::PointCloudStatePtr particles = tracker_->getParticles();
        if (particles) {
            for (const auto& point : particles->points) {
                particle_cloud->points.emplace_back(point.x, point.y, point.z);
            }          
        }
    }

    geometry_msgs::TransformStamped broadcastParticleTransform()
    {
        ParticleXYZRPY result = tracker_->getResult();
        Eigen::Affine3f transformation = tracker_->toEigenMatrix(result);
        
        //publish 3D sensor TF
        geometry_msgs::TransformStamped transformStamped;
        
        transformStamped.header.frame_id = "base_3D_sensor";
        transformStamped.child_frame_id = "base_target";
        transformStamped.transform.translation.x = transformation.translation()[0];
        transformStamped.transform.translation.y = transformation.translation()[1];
        transformStamped.transform.translation.z = transformation.translation()[2];
        tf2::Quaternion q;
        q.setRPY(0, 0, 0);
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        return transformStamped;
    }

    void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        CloudPtr cloud(new Cloud);
        CloudPtr cloud_filtered(new Cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 point_cloud2;
        pcl::PCLPointCloud2 point_cloud2_filtered; 
        geometry_msgs::TransformStamped transformStamped;
        CloudPtr target_cloud(new Cloud);

        // // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
        pcl::fromROSMsg (*cloud_msg, *cloud);
        // printf("%d\n", counter_);

        std::lock_guard<std::mutex> lock(mtx_);
        double start = pcl::getTime();
        FPS_CALC_BEGIN;
        cloud_pass_.reset(new Cloud);
        cloud_pass_downsampled_.reset(new Cloud);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        filterPassThrough(cloud, *cloud_pass_);
        if (counter_ < 50) {
            gridSample(cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
        }
        else if (counter_ == 50) {
            cloud_pass_downsampled_ = cloud_pass_;
            // CloudPtr target_cloud;
            if (use_convex_hull_) {
                planeSegmentation(cloud_pass_downsampled_, *coefficients, *inliers);
                if (inliers->indices.size() > 3) {
                    CloudPtr cloud_projected(new Cloud);
                    cloud_hull_.reset(new Cloud);
                    nonplane_cloud_.reset(new Cloud);

                    planeProjection(cloud_pass_downsampled_, *cloud_projected, coefficients);
                    convexHull(cloud_projected, *cloud_hull_, hull_vertices_);                   
                    extractNonPlanePoints(cloud_pass_downsampled_, cloud_hull_, *nonplane_cloud_);
                    target_cloud = nonplane_cloud_;
                    PCL_INFO("Got target!\n");
                }
                else {
                    PCL_WARN("cannot segment plane\n");
                }
            }
            else {
                PCL_WARN("without plane segmentation\n");
                CloudPtr source (new Cloud);
                if(pcl::io::loadPCDFile<PointType>("hydroflask_plane_filter.pcd", *source) == -1) {
                    PCL_ERROR("Couldn't load target file.\n");
                    exit(1);
                }
                target_cloud = source;
                PCL_WARN("I got the file in ig\n");
            }

            if (target_cloud != nullptr) {
                PCL_INFO("segmentation, please wait...\n");                
                std::vector<pcl::PointIndices> cluster_indices;
                gridSample(target_cloud, *target_cloud, 0.002);
                printf("down sample target size: %d\n", target_cloud->size());
                euclideanSegment(target_cloud, cluster_indices);
                if (!cluster_indices.empty()) {
                    // select the cluster to track
                    CloudPtr temp_cloud(new Cloud);
                    extractSegmentCluster(target_cloud, cluster_indices, 0, *temp_cloud);
                    printf("cluster size: %d\n", temp_cloud->size());
                    Eigen::Vector4f c;
                    pcl::compute3DCentroid<RefPointType>(*temp_cloud, c);
                    int segment_index = 0;
                    double segment_distance = c[0] * c[0] + c[1] * c[1];
                    for (std::size_t i = 1; i < cluster_indices.size(); i++) {
                        temp_cloud.reset(new Cloud);
                        extractSegmentCluster(target_cloud, cluster_indices, int(i), *temp_cloud);
                        pcl::compute3DCentroid<RefPointType>(*temp_cloud, c);
                        double distance = c[0] * c[0] + c[1] * c[1];
                        if (distance < segment_distance) {
                            segment_index = int(i);
                            segment_distance = distance;
                        }
                    }

                    segmented_cloud_.reset(new Cloud);
                    extractSegmentCluster(
                        target_cloud, cluster_indices, segment_index, *segmented_cloud_);
                    RefCloudPtr ref_cloud(new RefCloud);
                    ref_cloud = segmented_cloud_;
                    RefCloudPtr nonzero_ref(new RefCloud);
                    removeZeroPoints(ref_cloud, *nonzero_ref);

                    PCL_INFO("calculating cog\n");

                    RefCloudPtr transed_ref(new RefCloud);
                    pcl::compute3DCentroid<RefPointType>(*nonzero_ref, c);
                    Eigen::Affine3f trans = Eigen::Affine3f::Identity();
                    trans.translation().matrix() = Eigen::Vector3f(c[0], c[1], c[2]);
                    pcl::transformPointCloud<RefPointType>(
                        *nonzero_ref, *transed_ref, trans.inverse());
                    CloudPtr transed_ref_downsampled(new Cloud);
                    gridSample(transed_ref, *transed_ref_downsampled, downsampling_grid_size_);
                    tracker_->setReferenceCloud(transed_ref_downsampled);
                    tracker_->setTrans(trans);
                    reference_ = transed_ref;
                    tracker_->setMinIndices(ref_cloud->size() / 2);                   
                }
                else {
                    PCL_WARN("euclidean segmentation failed\n");
                }
            }
        }
        else {
            gridSampleApprox(cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
            tracking(cloud_pass_downsampled_);

            if (visualize_particles_) {
                drawParticles(particle_cloud);
            }
            transformStamped = broadcastParticleTransform();
        }
        
        new_cloud_ = true;
        double end = pcl::getTime();
        computation_time_ = end - start;
        FPS_CALC_END("computation");
        counter_++;

        /////////////////////////////////////////////////////////////////////////
        // // Convert to ROS data type
        pcl::toPCLPointCloud2(*cloud, point_cloud2);
        pcl::toPCLPointCloud2(*particle_cloud, point_cloud2_filtered);
        sensor_msgs::PointCloud2 output;
        sensor_msgs::PointCloud2 output_filtered;
        pcl_conversions::fromPCL(point_cloud2, output);
        pcl_conversions::fromPCL(point_cloud2_filtered, output_filtered);
        output.header.frame_id = "base_3D_sensor"; // TODO fill in header
        output_filtered.header.frame_id = "base_3D_sensor"; // TODO fill in header
        output.header.stamp = ros::Time::now();
        output_filtered.header.stamp = ros::Time::now();
        transformStamped.header.stamp = ros::Time::now();
    

        // /////////////////////////////////////////////////////////////////////////
        // // Publish the data
        pub.publish(output);
        pub2.publish(output_filtered);
        tfb.sendTransform(transformStamped);
        // marker_pub.publish(marker);
    }

};

int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "my_pcl_tutorial");

    bool use_convex_hull = true;
    bool visualize_non_downsample = false;
    bool visualize_particles = true;
    bool use_fixed = false;
    double downsampling_grid_size = 0.01;
    SegmentTracking<pcl::PointXYZRGB> segmentTracking(
                                            8,
                                            downsampling_grid_size,
                                            use_convex_hull,
                                            visualize_non_downsample,
                                            visualize_particles,
                                            use_fixed);

    
    ros::Rate loop_rate(100);

    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

}