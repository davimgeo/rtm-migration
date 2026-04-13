#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

class Config
{
public:
  // ===== General =====
  bool debug = false;
  std::string engine = "";

  // ===== RTM =====
  bool save_image = false;
  bool is_laplacian = false;
  bool is_gradient = false;

  // ===== Modeling =====
  float dh = 0.0f;
  int nb = 0;
  float factor = 0.0f;

  // ===== Seismogram =====
  std::string seismogram_mode = "generate";
  std::string load_seis_path = "";
  int nt = 0;
  float dt = 0.0f;
  int perc = 99;

  // ===== Model =====
  std::string model_mode = "create";
  std::string model_path = "";
  int nx_load = 0;
  int nz_load = 0;
  int nx = 0;
  int nz = 0;
  int* interfaces;
  float* value_interfaces;

  // ===== Geometry =====
  std::string geometry_mode = "load";
  int nx_geom = 0;
  int nz_geom = 0;
  float rec_depth = 0.0f;
  int* sources_create;
  float src_depth = 0.0f;
  float offset = 0.0f;
  int* src_create;
  bool save_create = false;
  int nsrc = 0;

  // ===== Wavelet =====
  float fmax = 0.0f;
  float tlag = 0.0f;

  // ===== Snapshots =====
  bool snap_num_nyquist = false;
  int snap_num = 0;

public:
  void manageModes()
  {
    if (model_mode == "LOAD" || model_mode == "load") {
      nx = nx_load;
      nz = nz_load;
    }
  }
};

#endif
