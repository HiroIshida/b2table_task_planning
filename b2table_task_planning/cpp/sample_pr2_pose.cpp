#include <random>
#include <Eigen/Dense>
#include <iostream>

typedef bool (*pred_func_t)(const double* arr);

struct Pr2PoseSampler {
  Pr2PoseSampler(
      double* region_size,
      double* region_center,
      double* table_size,
      double* table_center,
      pred_func_t ineq_cst_pred,
      int seed) : ineq_cst_pred(ineq_cst_pred) {
    this->region_size = Eigen::Vector2d(region_size[0], region_size[1]);
    this->region_center = Eigen::Vector2d(region_center[0], region_center[1]);
    this->half_table_size = Eigen::Vector2d(table_size[0], table_size[1]) * 0.5;
    this->table_center = Eigen::Vector2d(table_center[0], table_center[1]);
    this->generator = std::default_random_engine(seed);
  } 

  bool sample(double* reaching_pose, double* pose_out) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto reaching_pose_eigen = Eigen::Map<Eigen::Vector4d>(reaching_pose);
    for(int i = 0; i < 100; i++) {
      // double rn[3] = {distribution(this->generator), distribution(this->generator), distribution(this->generator)}; 
      double rn_pos[2] = {distribution(this->generator), distribution(this->generator)};
      double rn_yaw = distribution(this->generator); // yaw is -pi/4 to pi/4
      Eigen::Vector2d rn_eigen = Eigen::Map<Eigen::Vector2d>(rn_pos);
      Eigen::Vector2d pos_eigen = this->region_center + this->region_size.cwiseProduct(rn_eigen) - this->region_size * 0.5;
      double dist_from_reaching = (pos_eigen - reaching_pose_eigen.head<2>()).norm();
      if(dist_from_reaching < 0.8) {
        double sd = compute_signed_distance(pos_eigen);
        if(sd > 0.55 || sd < 0.0) {
          continue;
        }
        double yaw = reaching_pose_eigen[3] + rn_yaw * M_PI / 2.0 - M_PI / 4.0;
        pose_out[0] = pos_eigen.x();
        pose_out[1] = pos_eigen.y();
        pose_out[2] = fit_radian(yaw);
        if(ineq_cst_pred(pose_out)) {
          return true;
        }
      }
    }
    return false;
  }

  inline double compute_signed_distance(const Eigen::Vector2d& p) {
    Eigen::Vector2d p_relative = p - table_center;
    Eigen::Vector2d d = p_relative.cwiseAbs() - half_table_size;
    Eigen::Vector2d d_pos = d.cwiseMax(0.0);
    double part_out = d_pos.norm();
    double part_in = std::min(std::max(d.x(), d.y()), 0.0);
    return part_out + part_in;
  }

  inline double fit_radian(double theta) {
    return fmod(theta + M_PI, 2 * M_PI) - M_PI;
  }

  Eigen::Vector2d region_size;
  Eigen::Vector2d region_center;
  Eigen::Vector2d half_table_size;
  Eigen::Vector2d table_center;
  pred_func_t ineq_cst_pred;
  std::default_random_engine generator;
};


extern "C" {
  void* create_sampler(double* region_size, double* region_center, double* table_size, double* table_center, pred_func_t ineq_cst_pred, int seed) {
    Pr2PoseSampler* sampler = new Pr2PoseSampler(region_size, region_center, table_size, table_center, ineq_cst_pred, seed);
    return (void*)sampler;
  }

  bool sample_pose(void* sampler, double* reaching_pose, double* pose_out) {
    Pr2PoseSampler* s = (Pr2PoseSampler*)sampler;
    return s->sample(reaching_pose, pose_out);
  }
  
  void destroy_sampler(void* sampler) {
    Pr2PoseSampler* s = (Pr2PoseSampler*)sampler;
    delete s;
  }
}
