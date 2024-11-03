//#include "Pattern_Lookup.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable

// Define constants for various parameters
#define NUM_CONSE_FAST 9
#define POINTS_PER_CELL 10
#define FAST_THRES 7
#define RESOLUTION_X 1280
#define NUM_PARTS_X 24
#define NUM_PARTS_Y 14
#define MAX_KPTS_PER_GRID 10
#define BLOCKSIZE_H 7
#define NUM_SEARCH_X 1
#define NUM_SEARCH_Y 1

// Constant arrays for Hessian matrix calculations
__constant int hoofr_hessian_dxx[49] = {
    8,   15,  -7,  -34,  -7,   15,   8,    44,  85,  -41, -191, -41,  85,
    44,  125, 240, -117, -542, -117, 240,  125, 177, 340, -166, -768, -166,
    340, 177, 125, 240,  -117, -542, -117, 240, 125, 44,  85,   -41,  -191,
    -41, 85,  44,  8,    15,   -7,   -34,  -7,  15,  8};

__constant int hoofr_hessian_dyy[49] = {
    8,    44,  125, 177,  125,  44,   8,    15,  85,  240,  340,  240,  85,
    15,   -7,  -41, -117, -166, -117, -41,  -7,  -34, -191, -542, -768, -542,
    -191, -34, -7,  -41,  -117, -166, -117, -41, -7,  15,   85,   240,  340,
    240,  85,  15,  8,    44,   125,  177,  125, 44,  8};

__constant int hoofr_hessian_dxy[49] = {
    9,   35,  50,  0,    -50,  -35,  -9,   35,  133, 188, 0,    -188, -133,
    -35, 50,  188, 266,  0,    -266, -188, -50, 0,   0,   0,    0,    0,
    0,   0,   -50, -188, -266, 0,    266,  188, 50,  -35, -133, -188, 0,
    188, 133, 35,  -9,   -35,  -50,  0,    50,  35,  9};

// Structs for various data types used in the program
struct __attribute__((packed, aligned(16))) Image_Trace {
  short x;
  short y;
  int hess_score;
};

struct __attribute__((packed, aligned(32))) Keypoint_infos_channel {
  int x;
  int y;
  int hessian;
  int octave;
};

struct __attribute__((packed, aligned(8))) Image_distribution {
  int ref[NUM_PARTS_X * NUM_PARTS_Y];
  int num_points[NUM_PARTS_X * NUM_PARTS_Y];
};

struct __attribute__((packed, aligned(2))) Keypoint_cell {
  unsigned char cell_x;
  unsigned char cell_y;
};

struct __attribute__((packed, aligned(32))) Dmatch_QualityFactor {
  float distance;
  int queryIdx;
  int trainIdx;
  double qf;
};

struct __attribute__((packed, aligned(16))) Keypoint_infos {
  float x;
  float y;
  int octave;
};

struct __attribute__((packed, aligned(16))) PatternPoint {
  float x;     // x coordinate relative to center
  float y;     // y coordinate relative to center
  float sigma; // Gaussian smoothing sigma
};

struct __attribute__((packed, aligned(2))) DescriptionPair {
  uchar i; // index of the first point
  uchar j; // index of the second point
};

struct __attribute__((packed, aligned(16))) OrientationPair {
  uchar i;       // index of the first point
  uchar j;       // index of the second point
  int weight_dx; // dx/(norm_sq))*4096
  int weight_dy; // dy/(norm_sq))*4096
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////Kernel
/// Detection///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to compute the Hessian score for a given image region
int compute_hessian_score(__global unsigned char *restrict _img, int _jx,
                          int _jy) {
  int hes_score, img_val;
  int Dxx = 0, Dyy = 0, Dxy = 0;
  int half_size = BLOCKSIZE_H / 2;
  int id_img, id_img_ref, id_H;

  id_img_ref = (_jy - half_size) * RESOLUTION_X + (_jx - half_size);

  // Loop through the block to compute the Hessian score
  for (int qy = 0; qy < BLOCKSIZE_H; qy++) {
    id_img = id_img_ref + qy * RESOLUTION_X;
    id_H = qy * BLOCKSIZE_H;

    for (int qx = 0; qx < BLOCKSIZE_H; qx++) {
      img_val = (int)_img[(id_img + qx)];
      Dxx += img_val * hoofr_hessian_dxx[id_H + qx];
      Dyy += img_val * hoofr_hessian_dyy[id_H + qx];
      Dxy += img_val * hoofr_hessian_dxy[id_H + qx];
    }
  }

  // Calculate the Hessian score
  hes_score = ((Dxx * Dyy) - (Dxy * Dxy));
  return hes_score;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////Kernel
/// Description/////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NB_POINTS 49
#define NB_ORIENTATION 256
#define NB_ORIENTATION_PAIRS 40
#define NB_ORIENTATION_on_2PI 40.743665432

#define NB_ORIENTATION_MUL_NB_POINTS 12544
#define NB_PAIRS 256

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////Channel
/// OPENCL///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Struct for data communication between kernels
struct DC {
  unsigned short ptidx;
  unsigned short num_ktps_grid;
};

// Channels for communication between kernels
channel struct DC grid_ready_channel
    __attribute__((depth(NUM_PARTS_X * NUM_PARTS_Y)));

channel struct DC grid_HESSIAN_COMPUTE_ready_channel
    __attribute__((depth((int)(NUM_PARTS_X * NUM_PARTS_Y / 2 + 1))));
channel struct DC grid_calcul_ready_channel
    __attribute__((depth((int)(NUM_PARTS_X * NUM_PARTS_Y / 2 + 1))));

channel struct DC grid_HESSIAN_COMPUTE_ready_channel_2
    __attribute__((depth((int)(NUM_PARTS_X * NUM_PARTS_Y / 2 + 1))));
channel struct DC grid_calcul_ready_channel_2
    __attribute__((depth((int)(NUM_PARTS_X * NUM_PARTS_Y / 2 + 1))));

// Function to add a keypoint to the keypoint list
void Add_to_keypoint_list(__global struct Keypoint_infos_channel *ktp_info_list,
                          unsigned short *num_elements, int *hessian_min,
                          int *postition_min,
                          struct Keypoint_infos_channel ktp_info_,
                          int pos_ref) {
  int hess_min, pos_min;

  // Check if the keypoint can be added to the list
  if (((*num_elements) < POINTS_PER_CELL) ||
      (ktp_info_.hessian > (*hessian_min))) {

    if ((*num_elements) < POINTS_PER_CELL) {
      pos_min = (int)(*num_elements);
      (*num_elements)++;
    } else if (ktp_info_.hessian > (*hessian_min)) {
      pos_min = (*postition_min);
    }

    // Add the keypoint to the list
    ktp_info_list[pos_min + pos_ref] = ktp_info_;

  } else {
    return;
  }

  // Update the minimum Hessian score and position if the list is full
  if ((*num_elements) >= POINTS_PER_CELL) {
    hess_min = 1000000000;

#pragma unroll 1
    for (int i = 0; i < POINTS_PER_CELL; i++) {
      if (hess_min > ktp_info_list[i + pos_ref].hessian) {
        hess_min = ktp_info_list[i + pos_ref].hessian;
        pos_min = i;
      }
    }
    (*hessian_min) = hess_min;
    (*postition_min) = pos_min;
  }

  return;
}

// Kernel function for Hessian filtering
__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Hessian_Filtering_channel(
    __global struct Image_Trace *restrict img_trace,
    __global struct Keypoint_infos_channel *restrict ktps_grid_glob,
    __global unsigned short *restrict num_ktps_grid) {
  struct DC dc;
  int ptidx;

  bool valid;
  valid = false;
  while (!valid) {
    dc = read_channel_nb_intel(grid_HESSIAN_COMPUTE_ready_channel, &valid);
    if (!valid) {
      dc = read_channel_nb_intel(grid_HESSIAN_COMPUTE_ready_channel_2, &valid);
    }
  }

  ptidx = (int)(dc.ptidx);

  int pos_reff = POINTS_PER_CELL * ptidx;
  int hess_score;

  struct Keypoint_infos_channel ktp_info;
  int hess_min = 0, pos_min;
  unsigned short num_elements = 0;

  unsigned short limit = dc.num_ktps_grid;
  int pos_ref = ptidx * MAX_KPTS_PER_GRID;

#pragma unroll 1
  for (int i = 0; i < limit; i++) {
    hess_score = (int)img_trace[pos_ref + i].hess_score;

    if ((hess_score > hess_min) || (num_elements < POINTS_PER_CELL)) {
      ktp_info.x = (int)img_trace[pos_ref + i].x;
      ktp_info.y = (int)img_trace[pos_ref + i].y;
      ktp_info.hessian = hess_score;
      ktp_info.octave = 0;
      Add_to_keypoint_list(ktps_grid_glob, &num_elements, &hess_min, &pos_min,
                           ktp_info, pos_reff);
    }
  }

  dc.num_ktps_grid = num_elements;

  write_channel_intel(grid_ready_channel, dc);
}

// Kernel function for computing Hessian scores
__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Hessian_Compute_channel(
    __global unsigned char *restrict img,
    __global struct Image_Trace *restrict img_trace,
    __global unsigned short *restrict num_ktps_grid) {
  struct DC dc = read_channel_intel(grid_calcul_ready_channel);

  int ptidx = (int)(dc.ptidx);

  int jx, jy;

  unsigned short limit = dc.num_ktps_grid;
  int pos_ref = ptidx * MAX_KPTS_PER_GRID;

#pragma unroll 1
  for (int i = 0; i < limit; i++) {
    jx = (int)img_trace[pos_ref + i].x;
    jy = (int)img_trace[pos_ref + i].y;
    img_trace[pos_ref + i].hess_score = compute_hessian_score(img, jx, jy);
  }

  write_channel_intel(grid_HESSIAN_COMPUTE_ready_channel, dc);
}

// Kernel function for computing Hessian scores (second channel)
__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Hessian_Compute_channel_2(
    __global unsigned char *restrict img,
    __global struct Image_Trace *restrict img_trace,
    __global unsigned short *restrict num_ktps_grid) {
  struct DC dc = read_channel_intel(grid_calcul_ready_channel_2);

  int ptidx = (int)(dc.ptidx);

  int jx, jy;

  unsigned short limit = dc.num_ktps_grid;
  int pos_ref = ptidx * MAX_KPTS_PER_GRID;

#pragma unroll 1
  for (int i = 0; i < limit; i++) {
    jx = (int)img_trace[pos_ref + i].x;
    jy = (int)img_trace[pos_ref + i].y;
    img_trace[pos_ref + i].hess_score = compute_hessian_score(img, jx, jy);
  }

  write_channel_intel(grid_HESSIAN_COMPUTE_ready_channel_2, dc);
}

// Constant array for points index
__constant int points_index_yy[16] = {
    -3 * RESOLUTION_X + 0, -3 * RESOLUTION_X + 1, -2 * RESOLUTION_X + 2,
    -1 * RESOLUTION_X + 3, 0 * RESOLUTION_X + 3,  1 * RESOLUTION_X + 3,
    2 * RESOLUTION_X + 2,  3 * RESOLUTION_X + 1,  3 * RESOLUTION_X + 0,
    3 * RESOLUTION_X - 1,  2 * RESOLUTION_X - 2,  1 * RESOLUTION_X - 3,
    0 * RESOLUTION_X - 3,  -1 * RESOLUTION_X - 3, -2 * RESOLUTION_X - 2,
    -3 * RESOLUTION_X - 1};

// Function to test if a point is a FAST corner
int test_FAST_corner_2(int brightdark) {
  unsigned short fast16_d, fast16_b, fast8_d, fast8_b, test16_d, test16_b;

  fast16_d = brightdark & 0x0000FFFF;
  fast16_b = (brightdark >> 16) & 0x0000FFFF;

  fast8_d = ((fast16_d >> 8) & 0x00FF) | (fast16_d & 0x00FF);
  if (fast8_d == 0x00FF) {
    test16_d = fast16_d & 0xFF80;
    if (test16_d == 0xFF80)
      return 1;
    test16_d = fast16_d & 0x7FC0;
    if (test16_d == 0x7FC0)
      return 1;
    test16_d = fast16_d & 0x3FE0;
    if (test16_d == 0x3FE0)
      return 1;
    test16_d = fast16_d & 0x1FF0;
    if (test16_d == 0x1FF0)
      return 1;

    test16_d = fast16_d & 0x0FF8;
    if (test16_d == 0x0FF8)
      return 1;
    test16_d = fast16_d & 0x07FC;
    if (test16_d == 0x07FC)
      return 1;
    test16_d = fast16_d & 0x03FE;
    if (test16_d == 0x03FE)
      return 1;
    test16_d = fast16_d & 0x01FF;
    if (test16_d == 0x01FF)
      return 1;

    test16_d = fast16_d & 0x80FF;
    if (test16_d == 0x80FF)
      return 1;
    test16_d = fast16_d & 0xC07F;
    if (test16_d == 0xC07F)
      return 1;
    test16_d = fast16_d & 0xE03F;
    if (test16_d == 0xE03F)
      return 1;
    test16_d = fast16_d & 0xF01F;
    if (test16_d == 0xF01F)
      return 1;

    test16_d = fast16_d & 0xF80F;
    if (test16_d == 0xF80F)
      return 1;
    test16_d = fast16_d & 0xFC07;
    if (test16_d == 0xFC07)
      return 1;
    test16_d = fast16_d & 0xFE03;
    if (test16_d == 0xFE03)
      return 1;
    test16_d = fast16_d & 0xFF01;
    if (test16_d == 0xFF01)
      return 1;
  }

  fast8_b = ((fast16_b >> 8) & 0x00FF) | (fast16_b & 0x00FF);
  if (fast8_b == 0x00FF) {
    test16_b = fast16_b & 0xFF80;
    if (test16_b == 0xFF80)
      return 2;
    test16_b = fast16_b & 0x7FC0;
    if (test16_b == 0x7FC0)
      return 2;
    test16_b = fast16_b & 0x3FE0;
    if (test16_b == 0x3FE0)
      return 2;
    test16_b = fast16_b & 0x1FF0;
    if (test16_b == 0x1FF0)
      return 2;

    test16_b = fast16_b & 0x0FF8;
    if (test16_b == 0x0FF8)
      return 2;
    test16_b = fast16_b & 0x07FC;
    if (test16_b == 0x07FC)
      return 2;
    test16_b = fast16_b & 0x03FE;
    if (test16_b == 0x03FE)
      return 2;
    test16_b = fast16_b & 0x01FF;
    if (test16_b == 0x01FF)
      return 2;

    test16_b = fast16_b & 0x80FF;
    if (test16_b == 0x80FF)
      return 2;
    test16_b = fast16_b & 0xC07F;
    if (test16_b == 0xC07F)
      return 2;
    test16_b = fast16_b & 0xE03F;
    if (test16_b == 0xE03F)
      return 2;
    test16_b = fast16_b & 0xF01F;
    if (test16_b == 0xF01F)
      return 2;

    test16_b = fast16_b & 0xF80F;
    if (test16_b == 0xF80F)
      return 2;
    test16_b = fast16_b & 0xFC07;
    if (test16_b == 0xFC07)
      return 2;
    test16_b = fast16_b & 0xFE03;
    if (test16_b == 0xFE03)
      return 2;
    test16_b = fast16_b & 0xFF01;
    if (test16_b == 0xFF01)
      return 2;
  }

  return -1000000000;
}

// Kernel function for detecting keypoints
__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Detection_channel(__global unsigned char *restrict img,
                         __global struct Image_Trace *restrict img_trace,
                         __global short *restrict sub_img_coors,
                         __global unsigned short *restrict num_ktps_grid,
                         int reff)

{

  uchar fast8_1, fast8_2, fast8_3, fast8_4;
  uchar dark, bright;
  int hess_score, test_corner, jjy, jmcx, fast32;
  short p[2], m;
  struct DC dc;

  unsigned short num_ktps;
  int pos_ref;

  int ptidx_ref = get_global_id(0);
  int ptidx = ptidx_ref + reff;

  short ly, ry, lx, rx;
  ly = sub_img_coors[ptidx * 4 + 1];
  ry = sub_img_coors[ptidx * 4 + 3];
  lx = sub_img_coors[ptidx * 4 + 0];
  rx = sub_img_coors[ptidx * 4 + 2];

  num_ktps = 0;
  pos_ref = MAX_KPTS_PER_GRID * ptidx;

#pragma unroll 1
  for (short jy = ly; jy < ry; jy++) {
    jjy = ((int)jy) * RESOLUTION_X;
    for (short jx = lx; jx < rx; jx++) {
      jmcx = jjy + ((int)jx);

      m = (short)img[jmcx];
      fast8_1 = 0;
      fast8_2 = 0;
      fast8_3 = 0;
      fast8_4 = 0;
      hess_score = -1000000000;
      test_corner = 0;
      dark = 1;
      bright = 1;

#pragma unroll 1
      for (int i = 0; i < 8; i++) {
        uchar ones = 0x01 << i;

        p[0] = m - (short)img[jmcx + points_index_yy[i]];
        p[1] = m - (short)img[jmcx + points_index_yy[i + 8]];

        if (dark == 1) {
          if (p[0] > FAST_THRES) {
            fast8_1 = fast8_1 | ones;
          }
          if (p[1] > FAST_THRES) {
            fast8_2 = fast8_2 | ones;
          }

          if (((fast8_1 | fast8_2) & ones) == 0x00) {
            dark = 0;
          }
        }

        if (bright == 1) {
          if (p[0] < -FAST_THRES) {
            fast8_3 = fast8_3 | ones;
          }
          if (p[1] < -FAST_THRES) {
            fast8_4 = fast8_4 | ones;
          }

          if (((fast8_3 | fast8_4) & ones) == 0x00) {
            bright = 0;
          }
        }

        if ((dark == 0) && (bright == 0)) {
          break;
        }
      }

      if ((dark != 0) || (bright != 0)) {
        fast32 = (((int)fast8_1) << 24) | (((int)fast8_2) << 16) |
                 (((int)fast8_3) << 8) | (((int)fast8_4));
        test_corner = test_FAST_corner_2(fast32);
      }

      if (test_corner > 0) {
        img_trace[pos_ref + num_ktps].x = jx;
        img_trace[pos_ref + num_ktps].y = jy;
        num_ktps++;
        if (num_ktps >= MAX_KPTS_PER_GRID) {
          break;
        }
      }
    }
    if (num_ktps >= MAX_KPTS_PER_GRID) {
      break;
    }
  }
  dc.num_ktps_grid = num_ktps;
  dc.ptidx = (unsigned short)ptidx;
  write_channel_intel(grid_calcul_ready_channel, dc);

  return;
}

// Kernel function for detecting keypoints (second channel)
__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Detection_channel_2(__global unsigned char *restrict img,
                           __global struct Image_Trace *restrict img_trace,
                           __global short *restrict sub_img_coors,
                           __global unsigned short *restrict num_ktps_grid,
                           int reff)

{

  uchar fast8_1, fast8_2, fast8_3, fast8_4;
  uchar dark, bright;
  int hess_score, test_corner, jjy, jmcx, fast32;
  short p[2], m;
  struct DC dc;

  unsigned short num_ktps;
  int pos_ref;

  int ptidx_ref = get_global_id(0);
  int ptidx = ptidx_ref + reff;

  short ly, ry, lx, rx;
  ly = sub_img_coors[ptidx * 4 + 1];
  ry = sub_img_coors[ptidx * 4 + 3];
  lx = sub_img_coors[ptidx * 4 + 0];
  rx = sub_img_coors[ptidx * 4 + 2];

  num_ktps = 0;
  pos_ref = MAX_KPTS_PER_GRID * ptidx;

#pragma unroll 1
  for (short jy = ly; jy < ry; jy++) {
    jjy = ((int)jy) * RESOLUTION_X;
    for (short jx = lx; jx < rx; jx++) {
      jmcx = jjy + ((int)jx);

      m = (short)img[jmcx];
      fast8_1 = 0;
      fast8_2 = 0;
      fast8_3 = 0;
      fast8_4 = 0;
      hess_score = -1000000000;
      test_corner = 0;
      dark = 1;
      bright = 1;

#pragma unroll 1
      for (int i = 0; i < 8; i = i + 1) {
        uchar ones = 0x01 << i;

        p[0] = m - (short)img[jmcx + points_index_yy[i]];
        p[1] = m - (short)img[jmcx + points_index_yy[i + 8]];

        if (dark == 1) {
          if (p[0] > FAST_THRES) {
            fast8_1 = fast8_1 | ones;
          }
          if (p[1] > FAST_THRES) {
            fast8_2 = fast8_2 | ones;
          }

          if (((fast8_1 | fast8_2) & ones) == 0x00) {
            dark = 0;
          }
        }

        if (bright == 1) {
          if (p[0] < -FAST_THRES) {
            fast8_3 = fast8_3 | ones;
          }
          if (p[1] < -FAST_THRES) {
            fast8_4 = fast8_4 | ones;
          }

          if (((fast8_3 | fast8_4) & ones) == 0x00) {
            bright = 0;
          }
        }

        if ((dark == 0) && (bright == 0)) {
          break;
        }
      }

      if ((dark != 0) || (bright != 0)) {
        fast32 = (((int)fast8_1) << 24) | (((int)fast8_2) << 16) |
                 (((int)fast8_3) << 8) | (((int)fast8_4));
        test_corner = test_FAST_corner_2(fast32);
      }

      if (test_corner > 0) {
        img_trace[pos_ref + num_ktps].x = jx;
        img_trace[pos_ref + num_ktps].y = jy;
        num_ktps++;
        if (num_ktps >= MAX_KPTS_PER_GRID) {
          break;
        }
      }
    }
    if (num_ktps >= MAX_KPTS_PER_GRID) {
      break;
    }
  }

  dc.num_ktps_grid = num_ktps;
  dc.ptidx = (unsigned short)ptidx;
  write_channel_intel(grid_calcul_ready_channel_2, dc);

  return;
}

// Constant array for description pairs
__constant unsigned char DESCRIPTION_PAIRS[2 * NB_PAIRS] = {
    48, 40, 48, 41, 48, 42, 48, 43, 48, 44, 48, 45, 48, 46, 48, 47, 48, 32, 48,
    33, 48, 34, 48, 35, 48, 36, 48, 37, 48, 38, 48, 39, 48, 24, 48, 25, 48, 26,
    48, 27, 48, 28, 48, 29, 48, 30, 48, 31, 41, 40, 42, 41, 43, 42, 44, 43, 45,
    44, 46, 45, 47, 46, 47, 40, 40, 32, 40, 33, 41, 33, 41, 34, 42, 34, 42, 35,
    43, 35, 43, 36, 44, 36, 44, 37, 45, 37, 45, 38, 46, 38, 46, 39, 47, 39, 47,
    32, 40, 24, 40, 25, 40, 31, 41, 24, 41, 25, 41, 26, 42, 25, 42, 26, 42, 27,
    43, 26, 43, 27, 43, 28, 44, 27, 44, 28, 44, 29, 45, 28, 45, 29, 45, 30, 46,
    29, 46, 30, 46, 31, 47, 30, 47, 31, 47, 24, 40, 16, 40, 17, 41, 17, 41, 18,
    42, 18, 42, 19, 43, 19, 43, 20, 44, 20, 44, 21, 45, 21, 45, 22, 46, 22, 46,
    23, 47, 23, 47, 16, 17, 16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22,
    23, 16, 32, 31, 32, 24, 33, 24, 33, 25, 34, 25, 34, 26, 35, 26, 35, 27, 36,
    27, 36, 28, 37, 28, 37, 29, 38, 29, 38, 30, 39, 30, 39, 31, 32, 23, 32, 16,
    32, 17, 33, 16, 33, 17, 33, 18, 34, 17, 34, 18, 34, 19, 35, 18, 35, 19, 35,
    20, 36, 19, 36, 20, 36, 21, 37, 20, 37, 21, 37, 22, 38, 21, 38, 22, 38, 23,
    39, 22, 39, 23, 39, 16, 32, 15, 32, 8,  33, 8,  33, 9,  34, 9,  34, 10, 35,
    10, 35, 11, 36, 11, 36, 12, 37, 12, 37, 13, 38, 13, 38, 14, 39, 14, 39, 15,
    9,  8,  10, 9,  11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 15, 8,  24, 16, 24,
    17, 25, 17, 25, 18, 26, 18, 26, 19, 27, 19, 27, 20, 28, 20, 28, 21, 29, 21,
    29, 22, 30, 22, 30, 23, 31, 23, 31, 16, 24, 15, 24, 8,  24, 9,  25, 8,  25,
    9,  25, 10, 26, 9,  26, 10, 26, 11, 27, 10, 27, 11, 27, 12, 28, 11, 28, 12,
    28, 13, 29, 12, 29, 13, 29, 14, 30, 13, 30, 14, 30, 15, 31, 14, 31, 15, 31,
    8,  24, 0,  24, 1,  25, 1,  25, 2,  26, 2,  26, 3,  27, 3,  27, 4,  28, 4,
    28, 5,  29, 5,  29, 6,  30, 6,  30, 7,  31, 7,  31, 0,  1,  0,  2,  1,  3,
    2,  4,  3,  5,  4,  6,  5,  7,  6,  7,  0,  16, 15, 16, 8,  17, 8,  17, 9,
    18, 9,  18, 10, 19, 10, 19, 11, 20, 11, 20, 12, 21, 12, 21, 13, 22, 13, 22,
    14, 23, 14, 23, 15, 8,  0,  8,  1,  9,  1,  9,  2,  10, 2,  10, 3,  11, 3,
    11, 4,  12, 4,  12, 5,  13, 5,  13, 6,  14, 6,  14, 7,  15, 7,  15, 0};

__constant unsigned char ORIENTATION_PAIRS[2 * NB_ORIENTATION_PAIRS] = {
    0,  3,  0,  5,  1,  4,  1,  6,  2,  5,  2,  7,  3,  6,  4,  7,
    8,  11, 8,  13, 9,  12, 9,  14, 10, 13, 10, 15, 11, 14, 12, 15,
    16, 19, 16, 21, 17, 20, 17, 22, 18, 21, 18, 23, 19, 22, 20, 23,
    24, 27, 24, 29, 25, 28, 25, 30, 26, 29, 26, 31, 27, 30, 28, 31,
    32, 35, 32, 37, 33, 36, 33, 38, 34, 37, 34, 39, 35, 38, 36, 39};

__constant int ORIENTATION_WEIGHT[2 * NB_ORIENTATION_PAIRS] = {
    102,  -41, 102,  42,  102,  42,  42,   102, 42,   102,  -41, 102, -41,  102,
    -101, 42,  156,  0,   111,  111, 111,  111, 0,    156,  0,   156, -110, 111,
    -110, 111, -155, 0,   216,  -88, 216,  89,  216,  89,   89,  216, 89,   216,
    -88,  216, -88,  216, -215, 89,  369,  0,   261,  261,  261, 261, 0,    369,
    0,    369, -260, 261, -260, 261, -368, 0,   559,  -230, 559, 231, 559,  231,
    231,  559, 231,  559, -230, 559, -230, 559, -558, 231};

__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_assemble_keypoints(__global unsigned short *restrict num_ktps_grid) {
  struct DC dc;
  dc = read_channel_intel(grid_ready_channel);
  int ptidx_grid = (int)(dc.ptidx);
  num_ktps_grid[ptidx_grid] = dc.num_ktps_grid;

  ///////////

  return;
}

//################################################# Matching
// functions#################################################################""
float DescriptorDistance(__local unsigned int *a, __local unsigned int *b,
                         int id, int jd) {
  int dist;
  int dist_unit[8];
  unsigned int v[8];

  v[0] = a[id + 0] ^ b[jd + 0];
  v[1] = a[id + 1] ^ b[jd + 1];
  v[2] = a[id + 2] ^ b[jd + 2];
  v[3] = a[id + 3] ^ b[jd + 3];
  v[4] = a[id + 4] ^ b[jd + 4];
  v[5] = a[id + 5] ^ b[jd + 5];
  v[6] = a[id + 6] ^ b[jd + 6];
  v[7] = a[id + 7] ^ b[jd + 7];

  v[0] = v[0] - ((v[0] >> 1) & 0x55555555);
  v[1] = v[1] - ((v[1] >> 1) & 0x55555555);
  v[2] = v[2] - ((v[2] >> 1) & 0x55555555);
  v[3] = v[3] - ((v[3] >> 1) & 0x55555555);
  v[4] = v[4] - ((v[4] >> 1) & 0x55555555);
  v[5] = v[5] - ((v[5] >> 1) & 0x55555555);
  v[6] = v[6] - ((v[6] >> 1) & 0x55555555);
  v[7] = v[7] - ((v[7] >> 1) & 0x55555555);

  v[0] = (v[0] & 0x33333333) + ((v[0] >> 2) & 0x33333333);
  v[1] = (v[1] & 0x33333333) + ((v[1] >> 2) & 0x33333333);
  v[2] = (v[2] & 0x33333333) + ((v[2] >> 2) & 0x33333333);
  v[3] = (v[3] & 0x33333333) + ((v[3] >> 2) & 0x33333333);
  v[4] = (v[4] & 0x33333333) + ((v[4] >> 2) & 0x33333333);
  v[5] = (v[5] & 0x33333333) + ((v[5] >> 2) & 0x33333333);
  v[6] = (v[6] & 0x33333333) + ((v[6] >> 2) & 0x33333333);
  v[7] = (v[7] & 0x33333333) + ((v[7] >> 2) & 0x33333333);

  dist_unit[0] = (((v[0] + (v[0] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[1] = (((v[1] + (v[1] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[2] = (((v[2] + (v[2] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[3] = (((v[3] + (v[3] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[4] = (((v[4] + (v[4] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[5] = (((v[5] + (v[5] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[6] = (((v[6] + (v[6] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  dist_unit[7] = (((v[7] + (v[7] >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

  dist_unit[0] += dist_unit[1];
  dist_unit[2] += dist_unit[3];
  dist_unit[4] += dist_unit[5];
  dist_unit[6] += dist_unit[7];

  dist_unit[0] += dist_unit[2];
  dist_unit[4] += dist_unit[6];

  dist = dist_unit[0] + dist_unit[4];

  return (float)dist;
}

__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Matching_1(__global unsigned int *restrict current_desc,
                  __global struct Keypoint_cell *restrict kp_group_cell,
                  __global unsigned int *restrict prev_desc,
                  __global struct Image_distribution *restrict img_distribution,
                  __global struct Dmatch_QualityFactor *restrict dmatch_fpga,
                  int N_elements) {

  __local unsigned int
      current_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];
  __local unsigned int
      prev_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];

  for (int i = 0; i < 8 * N_elements / 4; i++) {
    current_desc_loc[i] = current_desc[i];
    prev_desc_loc[i] = prev_desc[i];
  }

  for (int i = 0; i < N_elements / 4; i++) {
    float dist;
    float dist_min = 1000000.0, dist_min2 = 1000000.0;
    int trainIdx_min = 0;

    int NX = (int)kp_group_cell[i].cell_x;
    int NY = (int)kp_group_cell[i].cell_y;
    for (int kx = 0; kx < 2 * NUM_SEARCH_X + 1; kx++) {
      for (int ky = 0; ky < 2 * NUM_SEARCH_Y + 1; ky++) {
        int nx = NX + kx - NUM_SEARCH_X;
        int ny = NY + ky - NUM_SEARCH_Y;

        if ((nx >= 0) && (ny >= 0)) {
          if ((nx <= NUM_PARTS_X - 1) && (ny <= NUM_PARTS_Y - 1)) {
            int ref_l = img_distribution[0].ref[ny * NUM_PARTS_X + nx];
            int ref_h =
                ref_l + img_distribution[0].num_points[ny * NUM_PARTS_X + nx];

            for (int j = ref_l; j < ref_h; j++) {
              dist = DescriptorDistance(prev_desc_loc, current_desc_loc, 8 * i,
                                        8 * j);
              if (dist < dist_min) {
                trainIdx_min = j;
                dist_min = dist;
              } else if (dist < dist_min2) {
                dist_min2 = dist;
              }
            }
          }
        }
      }
    }

    dmatch_fpga[i].queryIdx = i;
    dmatch_fpga[i].trainIdx = trainIdx_min;
    dmatch_fpga[i].distance = dist_min;
    dmatch_fpga[i].qf = dist_min / dist_min2;
  }
  return;
}

__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Matching_2(__global unsigned int *restrict current_desc,
                  __global struct Keypoint_cell *restrict kp_group_cell,
                  __global unsigned int *restrict prev_desc,
                  __global struct Image_distribution *restrict img_distribution,
                  __global struct Dmatch_QualityFactor *restrict dmatch_fpga,
                  int N_elements) {

  __local unsigned int
      current_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];
  __local unsigned int
      prev_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];

  for (int i = N_elements / 4; i < 8 * N_elements / 2; i++) {
    current_desc_loc[i] = current_desc[i];
    prev_desc_loc[i] = prev_desc[i];
  }

  for (int i = N_elements / 4; i < N_elements / 2; i++) {
    float dist;
    float dist_min = 1000000.0, dist_min2 = 1000000.0;
    int trainIdx_min = 0;

    int NX = (int)kp_group_cell[i].cell_x;
    int NY = (int)kp_group_cell[i].cell_y;
    for (int kx = 0; kx < 2 * NUM_SEARCH_X + 1; kx++) {
      for (int ky = 0; ky < 2 * NUM_SEARCH_Y + 1; ky++) {
        int nx = NX + kx - NUM_SEARCH_X;
        int ny = NY + ky - NUM_SEARCH_Y;

        if ((nx >= 0) && (ny >= 0)) {
          if ((nx <= NUM_PARTS_X - 1) && (ny <= NUM_PARTS_Y - 1)) {
            int ref_l = img_distribution[0].ref[ny * NUM_PARTS_X + nx];
            int ref_h =
                ref_l + img_distribution[0].num_points[ny * NUM_PARTS_X + nx];

            for (int j = ref_l; j < ref_h; j++) {
              dist = DescriptorDistance(prev_desc_loc, current_desc_loc, 8 * i,
                                        8 * j);
              if (dist < dist_min) {
                trainIdx_min = j;
                dist_min = dist;
              } else if (dist < dist_min2) {
                dist_min2 = dist;
              }
            }
          }
        }
      }
    }

    dmatch_fpga[i].queryIdx = i;
    dmatch_fpga[i].trainIdx = trainIdx_min;
    dmatch_fpga[i].distance = dist_min;
    dmatch_fpga[i].qf = dist_min / dist_min2;
  }
  return;
}

__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Matching_3(__global unsigned int *restrict current_desc,
                  __global struct Keypoint_cell *restrict kp_group_cell,
                  __global unsigned int *restrict prev_desc,
                  __global struct Image_distribution *restrict img_distribution,
                  __global struct Dmatch_QualityFactor *restrict dmatch_fpga,
                  int N_elements) {

  __local unsigned int
      current_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];
  __local unsigned int
      prev_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];

  for (int i = N_elements / 2; i < 8 * (3 * N_elements / 4); i++) {
    current_desc_loc[i] = current_desc[i];
    prev_desc_loc[i] = prev_desc[i];
  }

  for (int i = N_elements / 2; i < (3 * N_elements / 4); i++) {
    float dist;
    float dist_min = 1000000.0, dist_min2 = 1000000.0;
    int trainIdx_min = 0;

    int NX = (int)kp_group_cell[i].cell_x;
    int NY = (int)kp_group_cell[i].cell_y;
    for (int kx = 0; kx < 2 * NUM_SEARCH_X + 1; kx++) {
      for (int ky = 0; ky < 2 * NUM_SEARCH_Y + 1; ky++) {
        int nx = NX + kx - NUM_SEARCH_X;
        int ny = NY + ky - NUM_SEARCH_Y;

        if ((nx >= 0) && (ny >= 0)) {
          if ((nx <= NUM_PARTS_X - 1) && (ny <= NUM_PARTS_Y - 1)) {
            int ref_l = img_distribution[0].ref[ny * NUM_PARTS_X + nx];
            int ref_h =
                ref_l + img_distribution[0].num_points[ny * NUM_PARTS_X + nx];

            for (int j = ref_l; j < ref_h; j++) {
              dist = DescriptorDistance(prev_desc_loc, current_desc_loc, 8 * i,
                                        8 * j);
              if (dist < dist_min) {
                trainIdx_min = j;
                dist_min = dist;
              } else if (dist < dist_min2) {
                dist_min2 = dist;
              }
            }
          }
        }
      }
    }

    dmatch_fpga[i].queryIdx = i;
    dmatch_fpga[i].trainIdx = trainIdx_min;
    dmatch_fpga[i].distance = dist_min;
    dmatch_fpga[i].qf = dist_min / dist_min2;
  }
  return;
}

__attribute__((uses_global_work_offset(0))) __kernel void
Kernel_Matching_4(__global unsigned int *restrict current_desc,
                  __global struct Keypoint_cell *restrict kp_group_cell,
                  __global unsigned int *restrict prev_desc,
                  __global struct Image_distribution *restrict img_distribution,
                  __global struct Dmatch_QualityFactor *restrict dmatch_fpga,
                  int N_elements) {

  __local unsigned int
      current_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];
  __local unsigned int
      prev_desc_loc[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * 8 / 4];

  for (int i = (3 * N_elements / 4); i < 8 * N_elements; i++) {
    current_desc_loc[i] = current_desc[i];
    prev_desc_loc[i] = prev_desc[i];
  }

  for (int i = (3 * N_elements / 4); i < N_elements; i++) {
    float dist;
    float dist_min = 1000000.0, dist_min2 = 1000000.0;
    int trainIdx_min = 0;

    int NX = (int)kp_group_cell[i].cell_x;
    int NY = (int)kp_group_cell[i].cell_y;
    for (int kx = 0; kx < 2 * NUM_SEARCH_X + 1; kx++) {
      for (int ky = 0; ky < 2 * NUM_SEARCH_Y + 1; ky++) {
        int nx = NX + kx - NUM_SEARCH_X;
        int ny = NY + ky - NUM_SEARCH_Y;

        if ((nx >= 0) && (ny >= 0)) {
          if ((nx <= NUM_PARTS_X - 1) && (ny <= NUM_PARTS_Y - 1)) {
            int ref_l = img_distribution[0].ref[ny * NUM_PARTS_X + nx];
            int ref_h =
                ref_l + img_distribution[0].num_points[ny * NUM_PARTS_X + nx];

            for (int j = ref_l; j < ref_h; j++) {
              dist = DescriptorDistance(prev_desc_loc, current_desc_loc, 8 * i,
                                        8 * j);
              if (dist < dist_min) {
                trainIdx_min = j;
                dist_min = dist;
              } else if (dist < dist_min2) {
                dist_min2 = dist;
              }
            }
          }
        }
      }
    }

    dmatch_fpga[i].queryIdx = i;
    dmatch_fpga[i].trainIdx = trainIdx_min;
    dmatch_fpga[i].distance = dist_min;
    dmatch_fpga[i].qf = dist_min / dist_min2;
  }
  return;
}
