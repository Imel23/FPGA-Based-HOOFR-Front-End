// Include necessary headers
#include "host.h"
#include <sys/time.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"

// Use standard printf for debug output
#define MYPRINTF printf

// Constants for block sizes and grid dimensions
#define BLOCK_SIZE_DESC 1
#define NUM_SUBPARTS_X 1
#define NUM_SUBPARTS_Y 1
#define POINTS_PER_CELL 10
#define RESOLUTION_X 1280
#define RESOLUTION_Y 720
#define NUM_PARTS_X 24
#define NUM_PARTS_Y 14
#define NUM_SEARCH_X 1
#define NUM_SEARCH_Y 1
#define MAX_KPTS_PER_GRID 10
#define EDGE_THRESHOLD 31
#define _FAST_thres 7
#define STRING_BUFFER_LEN 1024
#define MAX_SOURCE_SIZE (0x100000)
#define NB_KEYPOINTS (NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL)

// Function prototypes for device info functions
void device_info_ulong(cl_device_id device, cl_device_info param, const char *name);
void device_info_uint(cl_device_id device, cl_device_info param, const char *name);
void device_info_bool(cl_device_id device, cl_device_info param, const char *name);
void device_info_string(cl_device_id device, cl_device_info param, const char *name);
void display_device_info(cl_device_id device);

// Structure to store image distribution across grid cells
struct __attribute__((packed, aligned(8))) Image_distribution
{
	int ref[NUM_PARTS_X * NUM_PARTS_Y];		   // Reference indices for keypoints
	int num_points[NUM_PARTS_X * NUM_PARTS_Y]; // Number of keypoints in each cell
};

// Structure to represent a cell coordinate in the grid
struct __attribute__((packed, aligned(2))) Keypoint_cell
{
	unsigned char cell_x; // X-coordinate of the cell
	unsigned char cell_y; // Y-coordinate of the cell
};

// Structure to store matching results with quality factor
struct __attribute__((packed, aligned(32))) Dmatch_QualityFactor
{
	float distance; // Distance between descriptors
	int queryIdx;	// Index of the query keypoint
	int trainIdx;	// Index of the train keypoint
	double qf;		// Quality factor of the match
};

// Structure to store keypoint information
struct __attribute__((packed, aligned(16))) Keypoint_infos
{
	float x;	// X-coordinate of the keypoint
	float y;	// Y-coordinate of the keypoint
	int octave; // Octave level of the keypoint
};

// Structure to store keypoint information in a channel
struct __attribute__((packed, aligned(32))) Keypoint_infos_channel
{
	int x;		 // X-coordinate of the keypoint
	int y;		 // Y-coordinate of the keypoint
	int hessian; // Hessian response value
	int octave;	 // Octave level of the keypoint
};

// Structure to store trace information of the image
struct __attribute__((packed, aligned(16))) Image_Trace
{
	short x;		// X-coordinate
	short y;		// Y-coordinate
	int hess_score; // Hessian score
};

using namespace aocl_utils;

// Number of pyramid levels
int _num_levels = 1;

// Pointer to the HOOFR feature extractor implementation
Ptr<HOOFR_Impl> hoofr_extractor;

// OpenCL runtime configuration variables
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id device;
cl_context context = NULL;

// Command queues
cl_command_queue command_queue[11];

cl_program program = NULL;

// OpenCL buffers
cl_mem buffer_patternLookup;
cl_mem buffer_descriptionPairs;
cl_mem buffer_orientationPairs;
cl_mem buffer_keypoint;
cl_mem buffer_description;
cl_mem buffer_description_prev;
cl_mem buffer_pointsValues;
cl_mem buffer_ThetaIdx;
cl_int N_POINTS;
cl_int N_ORIEN;
cl_int N_PAIRS;
cl_int N_ORIEN_PAIRS;

// OpenCL kernels
cl_kernel kernel_matching[5];
cl_kernel kernel_detection;
cl_kernel kernel_detection_channel[2];
cl_kernel kernel_hessian_compute_channel[2];
cl_kernel kernel_hessian_filtering_channel;
cl_kernel Kernel_assemble_keypoints;
cl_kernel kernel_matching_channel;

// C++ variables for storing keypoints and descriptions
Mat hoofr_description_left_previous;
Image_distribution img_distribution_left_previous;
Mat hoofr_description_left;
Image_distribution img_distribution_left;

// Function to release OpenCL resources
void cleanup()
{
	for (int i = 0; i < 5; i++)
	{
		if (kernel_matching[i])
		{
			clReleaseKernel(kernel_matching[i]);
		}
	}

	for (int i = 0; i < 11; i++)
	{
		if (command_queue[i])
		{
			clReleaseCommandQueue(command_queue[i]);
		}
	}

	if (program)
	{
		clReleaseProgram(program);
	}
	if (context)
	{
		clReleaseContext(context);
	}
}

// Function to initialize OpenCL resources
bool init_opencl()
{
	// Initialize HOOFR feature extractor with specified parameters
	hoofr_extractor = makePtr<HOOFR_Impl>(
		NB_KEYPOINTS, 1.200000048F, _num_levels, EDGE_THRESHOLD,
		0, 2, 0, EDGE_THRESHOLD, _FAST_thres);
	hoofr_extractor->orientationNormalized = false;
	hoofr_extractor->scaleNormalized = true;

	cl_int status, ret;

	printf("Initializing OpenCL\n");

	// Set current working directory to executable directory
	if (!setCwdToExeDir())
	{
		return false;
	}

	// ***************** FPGA Initialization ***************** //

	// Get the OpenCL platform
	platform = findPlatform("Intel");
	if (platform == NULL)
	{
		printf("ERROR: Unable to find Intel OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL devices
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
	if (num_devices != 1)
	{
		printf("Number of devices is not 1\n");
		return false;
	}

	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %d device(s)\n", num_devices);
	printf("  %s\n", getDeviceName(device).c_str());

	// Display device information
	display_device_info(device);

	// Create the OpenCL context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the program from binary file
	std::string binary_file = getBoardBinaryFile("../aoc_build/hoofr", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

	// Build the program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create command queues
	for (int i = 0; i < 11; i++)
	{
		command_queue[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue");
	}

	// Create OpenCL kernels
	kernel_detection_channel[0] = clCreateKernel(program, "Kernel_Detection_channel", &ret);
	kernel_detection_channel[1] = clCreateKernel(program, "Kernel_Detection_channel_2", &ret);

	kernel_hessian_compute_channel[0] = clCreateKernel(program, "Kernel_Hessian_Compute_channel", &ret);
	kernel_hessian_compute_channel[1] = clCreateKernel(program, "Kernel_Hessian_Compute_channel_2", &ret);

	kernel_hessian_filtering_channel = clCreateKernel(program, "Kernel_Hessian_Filtering_channel", &ret);

	Kernel_assemble_keypoints = clCreateKernel(program, "Kernel_assemble_keypoints", &ret);
	for (int i = 0; i < 4; i++)
	{
		kernel_matching[i] = clCreateKernel(program, ("Kernel_Matching_" + std::to_string(i + 1)).c_str(), &ret);
	}

	// Initialize constants for HOOFR descriptor
	N_POINTS = 49;
	N_ORIEN = 256; // Number of orientations
	N_PAIRS = 256;
	N_ORIEN_PAIRS = 40;

	// Create OpenCL buffers
	buffer_patternLookup = clCreateBuffer(context, CL_MEM_READ_ONLY,
										  sizeof(HOOFR_Impl::PatternPoint) * 8 * N_ORIEN * N_POINTS, NULL, &ret);

	buffer_keypoint = clCreateBuffer(context, CL_MEM_READ_ONLY,
									 sizeof(Keypoint_infos) * NB_KEYPOINTS, NULL, &ret);

	buffer_description = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										sizeof(unsigned char) * NB_KEYPOINTS * N_PAIRS / 8, NULL, &ret);

	buffer_pointsValues = clCreateBuffer(context, CL_MEM_READ_WRITE,
										 sizeof(unsigned char) * NB_KEYPOINTS * N_POINTS, NULL, &ret);

	buffer_ThetaIdx = clCreateBuffer(context, CL_MEM_READ_WRITE,
									 sizeof(int) * NB_KEYPOINTS, NULL, &ret);

	// Write pattern lookup data to device memory
	ret = clEnqueueWriteBuffer(command_queue[0], buffer_patternLookup, CL_TRUE, 0,
							   sizeof(HOOFR_Impl::PatternPoint) * 8 * N_ORIEN * N_POINTS,
							   &(hoofr_extractor->patternLookup[0]), 0, NULL, NULL);

	return true;
}

// Function to extract image grid and perform feature detection, description, and matching using OpenCL
void Image_Grid_Extraction_Channel(
	Ptr<HOOFR_Impl> &hoofr_extractor,
	Mat &img,
	Mat &depth,
	Mat &mask,
	vector<KeyPoint> &keypoints,
	Image_distribution &img_distribution,
	Mat &description,
	int num_levels,
	int num_features,
	int edge_threshold,
	uchar &need_matching,
	struct Dmatch_QualityFactor *dmatch_fpga,
	Image_distribution &prev_img_distribution,
	Mat &prev_desc)
{
	// Variables for timing
	struct timeval begin_l, end_l, begin_tt, end_tt;
	long elapsed_secs_l, elapsed_secs_tt;
	cl_int ret;

	// Start timing for detection
	gettimeofday(&begin_l, 0);

	// Clear previous keypoints and descriptions
	keypoints.clear();
	description.release();

	// Initialize variables for image pyramid
	vector<float> layerScale(num_levels);
	vector<Mat> imagePyr_(num_levels);
	vector<size_t> n_features_per_level_(num_levels);

	// Compute the number of desired features per scale
	float factor = 1.0f / 1.2f;
	float n_desired_features_per_scale = num_features * (1.0f - factor) / (1.0f - std::pow(factor, num_levels));

	size_t sum_n_features = 0;
	for (int level = 0; level < num_levels - 1; ++level)
	{
		n_features_per_level_[level] = cvRound(n_desired_features_per_scale);
		sum_n_features += n_features_per_level_[level];
		n_desired_features_per_scale *= factor;
	}
	n_features_per_level_[num_levels - 1] = num_features - sum_n_features;

	// Build image pyramid
	for (int level = 0; level < num_levels; level++)
	{
		layerScale[level] = (float)pow(1.2, level);
		float scale = 1.0f / layerScale[level];
		Size sz(cvRound(img.cols * scale), cvRound(img.rows * scale));

		if (level != 0)
		{
			if (level < 0)
			{
				resize(img, imagePyr_[level], sz, 0, 0, INTER_LINEAR);
			}
			else
			{
				resize(imagePyr_[level - 1], imagePyr_[level], sz, 0, 0, INTER_LINEAR);
			}
		}
		else
		{
			imagePyr_[level] = img;
		}
	}

	// End timing for pyramid construction
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("detection_pyramid_construction : %ld \n", elapsed_secs_l);

	// Prepare grid for keypoint detection
	vector<vector<KeyPoint>> keypoints_units(NUM_PARTS_X * NUM_PARTS_Y * num_levels);

	int dcount = 0;
	int FAST_threshold = hoofr_extractor->getFastThreshold();
	int patch_size = hoofr_extractor->getPatchSize();

	// Prepare sub-images for detection
	gettimeofday(&begin_l, 0);
	float coor_scale = (float)pow(1.2, 0);
	vector<short> sub_imgs(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4);

	int delta_x = (imagePyr_[0].cols - 2 * edge_threshold) / NUM_PARTS_X;
	int delta_y = (imagePyr_[0].rows - 2 * edge_threshold) / NUM_PARTS_Y;
	int sub_delta_x = delta_x / NUM_SUBPARTS_X;
	int sub_delta_y = delta_y / NUM_SUBPARTS_Y;

	for (int i = 0; i < NUM_PARTS_X * NUM_PARTS_Y; i++)
	{
		int ix = i % NUM_PARTS_X;
		int iy = i / NUM_PARTS_X;

		Point2i pt_tl, pt_br;
		pt_tl.x = (ix > 0) ? ix * delta_x + edge_threshold : edge_threshold;
		pt_tl.y = (iy > 0) ? iy * delta_y + edge_threshold : edge_threshold;

		pt_br.x = (ix < NUM_PARTS_X - 1) ? (ix + 1) * delta_x + edge_threshold : imagePyr_[0].cols - edge_threshold;
		pt_br.y = (iy < NUM_PARTS_Y - 1) ? (iy + 1) * delta_y + edge_threshold : imagePyr_[0].rows - edge_threshold;

		for (int j = 0; j < NUM_SUBPARTS_X * NUM_SUBPARTS_Y; j++)
		{
			int jx = j % NUM_SUBPARTS_X;
			int jy = j / NUM_SUBPARTS_X;

			sub_imgs[i * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4 + j * 4 + 0] = (short)(pt_tl.x + jx * sub_delta_x);
			sub_imgs[i * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4 + j * 4 + 1] = (short)(pt_tl.y + jy * sub_delta_y);
			sub_imgs[i * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4 + j * 4 + 2] = (short)((jx < NUM_SUBPARTS_X - 1) ? pt_tl.x + (jx + 1) * sub_delta_x : pt_br.x);
			sub_imgs[i * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4 + j * 4 + 3] = (short)((jy < NUM_SUBPARTS_Y - 1) ? pt_tl.y + (jy + 1) * sub_delta_y : pt_br.y);
		}
	}

	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("detection_preparation : %ld \n", elapsed_secs_l);

	// ********************************************************************* //
	// ******************** OpenCL Detection Setup ************************* //
	// ********************************************************************* //

	gettimeofday(&begin_tt, 0);

	Mat img_Integral;

	// Create OpenCL buffers
	cl_mem buffer_imgintegral;
	cl_mem buffer_description_channel;
	cl_mem buffer_dmatch_cl;
	cl_mem buffer_description_actuel;

	cl_int N_elements_des = NB_KEYPOINTS;
	cl_int img_intg_cols = img_Integral.cols;
	int block_size_cl_des = BLOCK_SIZE_DESC;

	buffer_description_channel = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
												sizeof(unsigned char) * NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL * N_PAIRS / 8, NULL, &ret);
	buffer_imgintegral = clCreateBuffer(context, CL_MEM_READ_ONLY,
										sizeof(int) * (img.rows + 1) * (img.cols + 1), NULL, &ret);

	// Detect keypoints using OpenCL
	int img_cols = imagePyr_[0].cols;
	int img_rows = imagePyr_[0].rows;
	int edge_thres = edge_threshold;
	cl_mem buffer_img;
	cl_mem buffer_mask;
	cl_mem buffer_depth;
	cl_mem buffer_img_trace;
	cl_mem buffer_sub_img_coor;
	cl_mem buffer_ktps_grid_glob;
	cl_mem buffer_num_ktps_grid;

	cl_int N_elements = (img_cols - 2 * edge_threshold) * (img_rows - 2 * edge_threshold);
	int block_size_cl = 16;

	buffer_img = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_cols * img_rows, NULL, &ret);
	buffer_sub_img_coor = clCreateBuffer(context, CL_MEM_READ_ONLY,
										 sizeof(short) * NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4, NULL, &ret);
	buffer_img_trace = clCreateBuffer(context, CL_MEM_READ_WRITE,
									  sizeof(Image_Trace) * MAX_KPTS_PER_GRID * NUM_PARTS_X * NUM_PARTS_Y, NULL, &ret);
	buffer_ktps_grid_glob = clCreateBuffer(context, CL_MEM_READ_WRITE,
										   sizeof(Keypoint_infos_channel) * NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL, NULL, &ret);
	buffer_num_ktps_grid = clCreateBuffer(context, CL_MEM_READ_WRITE,
										  sizeof(unsigned short) * NUM_PARTS_X * NUM_PARTS_Y, NULL, &ret);

	vector<unsigned char> zero_array(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * POINTS_PER_CELL, 0);
	vector<int> zero_int(NUM_PARTS_X * NUM_PARTS_Y * 8, 0);

	unsigned char *pointer_img = (unsigned char *)(&(imagePyr_[0].at<uchar>(0, 0)));
	unsigned char *pointer_mask = (unsigned char *)(&(mask.at<uchar>(0, 0)));
	uint16_t *pointer_depth = (uint16_t *)(&(depth.at<uchar>(0, 0)));
	short *pointer_sub_img_coor = (short *)(&(sub_imgs[0]));

	// Write image and sub-image coordinates to buffers
	gettimeofday(&begin_l, 0);
	ret = clEnqueueWriteBuffer(command_queue[0], buffer_img, CL_TRUE, 0,
							   sizeof(unsigned char) * img_cols * img_rows, pointer_img, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue[0], buffer_sub_img_coor, CL_TRUE, 0,
							   sizeof(short) * NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y * 4, pointer_sub_img_coor, 0, NULL, NULL);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("Write_buffer img and subimgcoor  : %ld \n", elapsed_secs_l);

	// Set kernel arguments for detection
	int reff = 0;
	int reff_2 = NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y / 2;

	ret = clSetKernelArg(kernel_detection_channel[0], 0, sizeof(unsigned char *), &buffer_img);
	ret = clSetKernelArg(kernel_detection_channel[0], 1, sizeof(Image_Trace *), &buffer_img_trace);
	ret = clSetKernelArg(kernel_detection_channel[0], 2, sizeof(short *), &buffer_sub_img_coor);
	ret = clSetKernelArg(kernel_detection_channel[0], 3, sizeof(unsigned short *), &buffer_num_ktps_grid);
	ret = clSetKernelArg(kernel_detection_channel[0], 4, sizeof(int), &reff);

	ret = clSetKernelArg(kernel_detection_channel[1], 0, sizeof(unsigned char *), &buffer_img);
	ret = clSetKernelArg(kernel_detection_channel[1], 1, sizeof(Image_Trace *), &buffer_img_trace);
	ret = clSetKernelArg(kernel_detection_channel[1], 2, sizeof(short *), &buffer_sub_img_coor);
	ret = clSetKernelArg(kernel_detection_channel[1], 3, sizeof(unsigned short *), &buffer_num_ktps_grid);
	ret = clSetKernelArg(kernel_detection_channel[1], 4, sizeof(int), &reff_2);

	ret = clSetKernelArg(kernel_hessian_compute_channel[0], 0, sizeof(unsigned char *), &buffer_img);
	ret = clSetKernelArg(kernel_hessian_compute_channel[0], 1, sizeof(Image_Trace *), &buffer_img_trace);
	ret = clSetKernelArg(kernel_hessian_compute_channel[0], 2, sizeof(unsigned short *), &buffer_num_ktps_grid);

	ret = clSetKernelArg(kernel_hessian_compute_channel[1], 0, sizeof(unsigned char *), &buffer_img);
	ret = clSetKernelArg(kernel_hessian_compute_channel[1], 1, sizeof(Image_Trace *), &buffer_img_trace);
	ret = clSetKernelArg(kernel_hessian_compute_channel[1], 2, sizeof(unsigned short *), &buffer_num_ktps_grid);

	ret = clSetKernelArg(kernel_hessian_filtering_channel, 0, sizeof(Image_Trace *), &buffer_img_trace);
	ret = clSetKernelArg(kernel_hessian_filtering_channel, 1, sizeof(Keypoint_infos_channel *), &buffer_ktps_grid_glob);
	ret = clSetKernelArg(kernel_hessian_filtering_channel, 2, sizeof(unsigned short *), &buffer_num_ktps_grid);

	size_t global_sizes_detection[] = {(size_t)(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y / 2), 1, 1};
	size_t global_sizes_detection_2[] = {(size_t)(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y - reff_2), 1, 1};

	size_t global_sizes_hessian_compute[] = {(size_t)(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y / 2), 1, 1};
	size_t global_sizes_hessian_compute_2[] = {(size_t)(NUM_PARTS_X * NUM_PARTS_Y * NUM_SUBPARTS_X * NUM_SUBPARTS_Y - reff_2), 1, 1};

	size_t global_sizes_hessian_filtering[] = {(size_t)(NUM_PARTS_X * NUM_PARTS_Y), 1, 1};

	ret = clSetKernelArg(Kernel_assemble_keypoints, 0, sizeof(unsigned short *), &buffer_num_ktps_grid);

	size_t workitems_des = (size_t)((N_elements_des + block_size_cl_des - 1) / block_size_cl_des * block_size_cl_des);
	size_t global_sizes_des[] = {workitems_des, (size_t)(N_PAIRS / 8), (size_t)(1)};
	size_t local_size_des[] = {(size_t)block_size_cl_des, (size_t)(N_PAIRS / 8), (size_t)(1)};

	global_sizes_des[0] = NUM_PARTS_X * NUM_PARTS_Y;
	global_sizes_des[1] = (size_t)((N_POINTS - 9 + 1 - 1) / 1 * 1);
	local_size_des[0] = (size_t)block_size_cl_des;
	local_size_des[1] = (size_t)1;

	// Compute integral image
	gettimeofday(&begin_l, 0);
	integral(img, img_Integral);
	int *pointer_img_Integral = (&(img_Integral.at<int>(0, 0)));
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("integral_image_compute : %ld \n", elapsed_secs_l);

	// Write integral image to buffer
	gettimeofday(&begin_l, 0);
	ret = clEnqueueWriteBuffer(command_queue[9], buffer_imgintegral, CL_TRUE, 0,
							   sizeof(int) * img_Integral.rows * img_Integral.cols, pointer_img_Integral, 0, NULL, NULL);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("integral_image_transfer : %ld \n", elapsed_secs_l);

	// Execute detection kernels
	gettimeofday(&begin_l, 0);
	ret = clEnqueueNDRangeKernel(command_queue[1], kernel_detection_channel[0], 1, NULL, global_sizes_detection, NULL, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(command_queue[2], kernel_detection_channel[1], 1, NULL, global_sizes_detection_2, NULL, 0, NULL, NULL);
	clFinish(command_queue[1]);
	clFinish(command_queue[2]);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - begin_l.tv_usec;
	MYPRINTF("kernel_detection_channel: %ld \n", elapsed_secs_l);

	gettimeofday(&begin_l, 0);
	ret = clEnqueueNDRangeKernel(command_queue[3], kernel_hessian_compute_channel[0], 1, NULL, global_sizes_hessian_compute, NULL, 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(command_queue[4], kernel_hessian_compute_channel[1], 1, NULL, global_sizes_hessian_compute_2, NULL, 0, NULL, NULL);
	clFinish(command_queue[3]);
	clFinish(command_queue[4]);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	MYPRINTF("kernel_hessian_compute_channel: %ld \n", elapsed_secs_l);

	gettimeofday(&begin_l, 0);
	ret = clEnqueueNDRangeKernel(command_queue[5], kernel_hessian_filtering_channel, 1, NULL, global_sizes_hessian_filtering, NULL, 0, NULL, NULL);
	clFinish(command_queue[5]);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	MYPRINTF("kernel_hessian_filtering_channel: %ld \n", elapsed_secs_l);

	// ********************************************************************* //
	// ******************* End Detection OpenCL **************************** //
	// ********************************************************************* //

	// Assemble keypoints
	gettimeofday(&begin_l, 0);
	ret = clEnqueueNDRangeKernel(command_queue[9], Kernel_assemble_keypoints, 1, NULL, global_sizes_des, NULL, 0, NULL, NULL);
	clFinish(command_queue[9]);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	MYPRINTF("Kernel_assemble_keypoints : %ld \n", elapsed_secs_l);

	clFinish(command_queue[9]);

	// Read back keypoints and descriptors
	Mat desc(NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL, N_PAIRS / 8, CV_8UC1);
	unsigned short num_ktps_grid[NUM_PARTS_X * NUM_PARTS_Y];
	Keypoint_infos_channel ktps_infos_channel[NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL];

	unsigned char *pointer_description = (unsigned char *)(&(desc.at<uchar>(0, 0)));
	gettimeofday(&begin_l, 0);
	ret = clEnqueueReadBuffer(command_queue[0], buffer_num_ktps_grid, CL_TRUE, 0,
							  sizeof(unsigned short) * NUM_PARTS_X * NUM_PARTS_Y, num_ktps_grid, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue[0], buffer_ktps_grid_glob, CL_TRUE, 0,
							  sizeof(Keypoint_infos_channel) * NUM_PARTS_X * NUM_PARTS_Y * POINTS_PER_CELL, ktps_infos_channel, 0, NULL, NULL);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	MYPRINTF("Assemble keypoints from buffer : %ld \n", elapsed_secs_l);

	// Transform keypoints to C++ form
	gettimeofday(&begin_l, 0);
	int num_kpts_total = 0;
	KeyPoint ktp_unit;
	for (int i = 0; i < NUM_PARTS_X * NUM_PARTS_Y; i++)
	{
		img_distribution.ref[i] = (int)keypoints.size();

		if (num_ktps_grid[i] > 0)
		{
			num_kpts_total += (int)num_ktps_grid[i];
			for (int j = 0; j < (int)num_ktps_grid[i]; j++)
			{
				ktp_unit.pt.x = ktps_infos_channel[POINTS_PER_CELL * i + j].x;
				ktp_unit.pt.y = ktps_infos_channel[POINTS_PER_CELL * i + j].y;
				ktp_unit.octave = ktps_infos_channel[POINTS_PER_CELL * i + j].octave;
				keypoints.push_back(ktp_unit);
			}
		}

		img_distribution.num_points[i] = (int)keypoints.size() - img_distribution.ref[i];
	}
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	MYPRINTF("Transform to C++ form : %ld \n", elapsed_secs_l);

	// Compute descriptors
	gettimeofday(&begin_l, 0);
	if ((int)keypoints.size() <= 0)
	{
		return;
	}
	hoofr_extractor->compute(img, keypoints, description);
	gettimeofday(&end_l, 0);
	elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
	printf("Description (ms): %f \n", elapsed_secs_l / 1000.0);

	// Matching phase
	if (need_matching == 1)
	{
		vector<Keypoint_cell> kpts_cell_left_previous_group;
		kpts_cell_left_previous_group.clear();
		Keypoint_cell k_cell;
		cl_mem buffer_img_distribution;
		cl_mem buffer_dmatch_cl;
		cl_mem buffer_kp_group_cell;
		cl_mem buffer_description;
		cl_mem buffer_description_prev;

		for (int j = 0; j < NUM_PARTS_X * NUM_PARTS_Y; j++)
		{
			k_cell.cell_x = (unsigned char)(j % NUM_PARTS_X);
			k_cell.cell_y = (unsigned char)(j / NUM_PARTS_X);
			for (int k = prev_img_distribution.ref[j]; k < prev_img_distribution.ref[j] + prev_img_distribution.num_points[j]; k++)
			{
				kpts_cell_left_previous_group.push_back(k_cell);
			}
		}
		int cell_group_size = kpts_cell_left_previous_group.size();

		buffer_kp_group_cell = clCreateBuffer(context, CL_MEM_READ_ONLY,
											  sizeof(struct Keypoint_cell) * cell_group_size, NULL, &ret);
		buffer_img_distribution = clCreateBuffer(context, CL_MEM_READ_ONLY,
												 sizeof(struct Image_distribution) * 1, NULL, &ret);
		buffer_dmatch_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										  sizeof(struct Dmatch_QualityFactor) * prev_desc.rows, NULL, &ret);
		buffer_description = clCreateBuffer(context, CL_MEM_READ_ONLY,
											sizeof(unsigned int) * description.rows * N_PAIRS / 32, NULL, &ret);
		buffer_description_prev = clCreateBuffer(context, CL_MEM_READ_ONLY,
												 sizeof(unsigned int) * prev_desc.rows * N_PAIRS / 32, NULL, &ret);

		unsigned int *pointer_current_descriptor = (unsigned int *)(&(description.at<uchar>(0, 0)));
		unsigned int *pointer_previous_descriptor = (unsigned int *)(&(prev_desc.at<uchar>(0, 0)));
		gettimeofday(&begin_l, 0);
		ret = clEnqueueWriteBuffer(command_queue[7], buffer_kp_group_cell, CL_TRUE, 0,
								   sizeof(struct Keypoint_cell) * cell_group_size, &(kpts_cell_left_previous_group[0]), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue[7], buffer_img_distribution, CL_TRUE, 0,
								   sizeof(struct Image_distribution) * 1, &(img_distribution), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue[7], buffer_description, CL_TRUE, 0,
								   sizeof(unsigned int) * description.rows * N_PAIRS / 32, pointer_current_descriptor, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue[7], buffer_description_prev, CL_TRUE, 0,
								   sizeof(unsigned int) * prev_desc.rows * N_PAIRS / 32, pointer_previous_descriptor, 0, NULL, NULL);
		gettimeofday(&end_l, 0);
		elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
		MYPRINTF("Write to buffer for Matching : %ld \n", elapsed_secs_l);

		// Set kernel arguments for Matching
		for (int i = 0; i < 4; i++)
		{
			ret = clSetKernelArg(kernel_matching[i], 0, sizeof(unsigned int *), &buffer_description);
			ret = clSetKernelArg(kernel_matching[i], 1, sizeof(struct Keypoint_cell *), &buffer_kp_group_cell);
			ret = clSetKernelArg(kernel_matching[i], 2, sizeof(unsigned int *), &buffer_description_prev);
			ret = clSetKernelArg(kernel_matching[i], 3, sizeof(struct Image_distribution *), &buffer_img_distribution);
			ret = clSetKernelArg(kernel_matching[i], 4, sizeof(struct Dmatch_QualityFactor *), &buffer_dmatch_cl);
			ret = clSetKernelArg(kernel_matching[i], 5, sizeof(cl_int), &prev_desc.rows);
		}

		gettimeofday(&begin_l, 0);

		for (int i = 0; i < 4; i++)
		{
			ret = clEnqueueTask(command_queue[7 + i], kernel_matching[i], 0, NULL, NULL);
		}

		for (int i = 7; i < 11; i++)
		{
			clFinish(command_queue[i]);
		}

		gettimeofday(&end_l, 0);
		elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
		MYPRINTF("Matching : %ld \n", elapsed_secs_l);

		gettimeofday(&begin_l, 0);
		ret = clEnqueueReadBuffer(command_queue[7], buffer_dmatch_cl, CL_TRUE, 0,
								  sizeof(struct Dmatch_QualityFactor) * prev_desc.rows, &dmatch_fpga[0], 0, NULL, NULL);
		gettimeofday(&end_l, 0);
		elapsed_secs_l = (end_l.tv_sec - begin_l.tv_sec) * 1000000 + end_l.tv_usec - end_l.tv_usec;
		MYPRINTF("Read from buffer Matches : %ld \n", elapsed_secs_l);
	}

	gettimeofday(&end_tt, 0);
	elapsed_secs_tt = (end_tt.tv_sec - begin_tt.tv_sec) * 1000000 + end_tt.tv_usec - begin_tt.tv_usec;
	MYPRINTF("Detection + Description + Matching: %ld \n", elapsed_secs_tt);

	// Release OpenCL buffers
	ret = clReleaseMemObject(buffer_img);
	ret = clReleaseMemObject(buffer_sub_img_coor);
	ret = clReleaseMemObject(buffer_ktps_grid_glob);
	ret = clReleaseMemObject(buffer_num_ktps_grid);
	ret = clReleaseMemObject(buffer_imgintegral);

	MYPRINTF("num_kpts_total : %d \n", num_kpts_total);
	MYPRINTF("des_rows_cols : %d %d \n", description.rows, description.cols);
}

// Main function
int main()
{
	struct timeval tv_start, tv_stop, tv1, tv0;
	long tv_elapsed, elapsed;

	// Initialize OpenCL
	init_opencl();

	Mat image_left, image_left_previous;
	Mat image_left_preresize, image_left_previous_preresize;
	vector<KeyPoint> hoofr_keypoints_left, hoofr_keypoints_left_previous;
	Image_distribution img_distribution, img_distribution_prev;
	Mat desc, desc_prev;

	// Load images
	image_left_preresize = imread("../images/frame846.png", 0);
	image_left_previous_preresize = imread("../images/frame847.png", 0);

	Mat mask = imread("../images/mask846.png", 0);
	Mat mask_prev = imread("../images/mask847.png", 0);

	Mat depth = imread("../images/depth846.png", 0);
	Mat depth_prev = imread("../images/depth847.png", 0);

	// Resize images to desired resolution
	resize(image_left_preresize, image_left, cv::Size(RESOLUTION_X, RESOLUTION_Y));
	resize(image_left_previous_preresize, image_left_previous, cv::Size(RESOLUTION_X, RESOLUTION_Y));

	MYPRINTF("\n \n \n \n \n \n");
	Mat description_actuel;
	Image_distribution img_distribution_actuel;
	uchar need_matching = 0;
	vector<Dmatch_QualityFactor> matches;
	Image_Grid_Extraction_Channel(hoofr_extractor, image_left_previous, depth_prev, mask_prev,
								  hoofr_keypoints_left_previous, img_distribution_prev, desc_prev,
								  hoofr_extractor->getNLevels(), NB_KEYPOINTS, hoofr_extractor->getEdgeThreshold(),
								  need_matching, &(matches[0]), img_distribution_prev, desc_prev);
	matches.clear();
	matches.resize(desc_prev.rows);

	need_matching = 1;
	Image_Grid_Extraction_Channel(hoofr_extractor, image_left, depth, mask,
								  hoofr_keypoints_left, img_distribution, desc,
								  hoofr_extractor->getNLevels(), NB_KEYPOINTS, hoofr_extractor->getEdgeThreshold(),
								  need_matching, &(matches[0]), img_distribution_prev, desc_prev);

	// Prepare matches for plotting
	vector<DMatch> plot_matches;
	plot_matches.clear();
	DMatch match_t;
	Mat img_out_matches;
	for (int i = 0; i < matches.size(); i++)
	{
		match_t.distance = matches[i].distance;
		match_t.queryIdx = matches[i].queryIdx;
		match_t.trainIdx = matches[i].trainIdx;
		if (matches[i].distance <= 20)
		{
			plot_matches.push_back(match_t);
		}
	}
	cout << "Recall : " << plot_matches.size() * 100.0 / matches.size() << endl;
	cout << "1-Precision : " << (matches.size() - plot_matches.size()) * 100.0 / matches.size() << endl;
	drawMatches(image_left_previous, hoofr_keypoints_left_previous, image_left, hoofr_keypoints_left,
				plot_matches, img_out_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Matches Keypoints", img_out_matches);
	waitKey(0);

	return 0;
}

// Helper functions to display device information
void device_info_ulong(cl_device_id device, cl_device_info param, const char *name)
{
	cl_ulong a;
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
	printf("%-40s = %lu\n", name, a);
}

void device_info_uint(cl_device_id device, cl_device_info param, const char *name)
{
	cl_uint a;
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
	printf("%-40s = %u\n", name, a);
}

void device_info_bool(cl_device_id device, cl_device_info param, const char *name)
{
	cl_bool a;
	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
	printf("%-40s = %s\n", name, (a ? "true" : "false"));
}

void device_info_string(cl_device_id device, cl_device_info param, const char *name)
{
	char a[STRING_BUFFER_LEN];
	clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
	printf("%-40s = %s\n", name, a);
}

void display_device_info(cl_device_id device)
{
	printf("Querying device for info:\n");
	printf("========================\n");
	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
	device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

	cl_command_queue_properties ccp;
	clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
	printf("%-40s = %s\n", "Command queue out of order?", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
	printf("%-40s = %s\n", "Command queue profiling enabled?", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
}
