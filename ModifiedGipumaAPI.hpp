#ifndef VO3DRP_MODIFIEDGIPUMA_HPP
#define VO3DRP_MODIFIEDGIPUMA_HPP

#include <dirent.h>
#include <sys/stat.h>
#include "displayUtils.h"
#include "main.h"
#include "algorithmparameters.h"
#include "gipuma.h"
#include "globalstate.h"

static void modifiedSelectViews(CameraParameters &cameraParams,
                                int imgWidth,
                                int imgHeight,
                                AlgorithmParameters &algParams)
{
  vector<Camera> &cameras = cameraParams.cameras;
  Camera ref = cameras[cameraParams.idRef];

  int x = imgWidth / 2;
  int y = imgHeight / 2;

  cameraParams.viewSelectionSubset.clear();

  Vec3f viewVectorRef = getViewVector(ref, x, y);

  // TODO hardcoded value makes it a parameter
  float minimum_angle_degree = algParams.min_angle;
  float maximum_angle_degree = algParams.max_angle;

  unsigned int maximum_view = algParams.max_views;
  float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
  float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
  float min_depth = 9999;
  float max_depth = 0;
  if (algParams.viewSelection)
  {
    printf(
        "Accepting intersection angle of central rays from %f to %f degrees, use --min_angle=<angle> and --max_angle=<angle> to modify them\n",
        minimum_angle_degree,
        maximum_angle_degree);
  }
  for (size_t i = 1; i < cameras.size(); i++)
  {
    //if ( !algParams.viewSelection ) { //select all views, dont perform selection
    //cameraParams.viewSelectionSubset.push_back ( i );
    //continue;
    //}

    Vec3f vec = getViewVector(cameras[i], x, y);

    float baseline = norm(cameras[0].C, cameras[i].C);
    float angle = getAngle(viewVectorRef, vec);
    if (angle > minimum_angle_radians &&
        angle
        < maximum_angle_radians) //0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
    {
      if (algParams.viewSelection)
      {
        cameraParams.viewSelectionSubset.push_back(i);
      }
      float min_range = (baseline / 2.0f) / sin(maximum_angle_radians / 2.0f);
      float max_range = (baseline / 2.0f) / sin(minimum_angle_radians / 2.0f);
      min_depth = std::min(min_range, min_depth);
      max_depth = std::max(max_range, max_depth);
      printf("Min max ranges are %f %f\n", min_range, max_range);
      printf("Min max depth are %f %f\n", min_depth, max_depth);
    }
  }

  if (algParams.depthMin == -1)
  {
    algParams.depthMin = min_depth;
  }
  if (algParams.depthMax == -1)
  {
    algParams.depthMax = max_depth;
  }
  //if (!algParams.viewSelection)
  {
    cameraParams.viewSelectionSubset.clear();
    for (size_t i = 0; i < algParams.max_views/*cameras.size()*/; i++)
    {
      cameraParams.viewSelectionSubset.push_back(i);
    }
  }
  if (cameraParams.viewSelectionSubset.size() >= maximum_view)
  {
    printf("Too many cameras ");//, randomly selecting only %d of them (modify with --max_views=<number>)\n", maximum_view);
    // todo: find better way for view selection
  }
}

static void delTexture(int num, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
  for (int i = 0; i < num; i++)
  {
    cudaFreeArray(cuArray[i]);
    cudaDestroyTextureObject(texs[i]);
  }
}

static void addImageToTextureFloatColor(vector<Mat> &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
  for (size_t i = 0; i < imgs.size(); i++)
  {
    int rows = imgs[i].rows;
    int cols = imgs[i].cols;
    // Create channel with floating point type
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // Allocate array with correct size and number of channels
    //cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray[i],
                                    &channelDesc,
                                    cols,
                                    rows));

    checkCudaErrors (cudaMemcpy2DToArray(cuArray[i],
                                         0,
                                         0,
                                         imgs[i].ptr<float>(),
                                         imgs[i].step[0],
                                         cols * sizeof(float) * 4,
                                         rows,
                                         cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray[i];

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    //cudaTextureObject_t &texObj = texs[i];
    checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
  }
  return;
}

/* get camera parameters (e.g. projection matrices) from file
 * Input:  inputFiles  - pathes to calibration files
 *         scaleFactor - if image was rescaled we need to adapt calibration matrix K accordingly
 * Output: camera parameters
 */
CameraParameters modifiedGetCameraParameters(CameraParameters_cu &cpc,
                                             std::vector<cv::Matx34f> pMatrices,
                                             float scaleFactor = 1.0f,
                                             bool transformP = true)
{

  CameraParameters params;
  //get projection matrices

  Mat_<float> KMaros = Mat::eye(3, 3, CV_32F);
  KMaros(0, 0) = 8066.0;
  KMaros(1, 1) = 8066.0;
  KMaros(0, 2) = 2807.5;
  KMaros(1, 2) = 1871.5;

  // transfer P matrices to gipuma format

  unsigned long numCams = pMatrices.size();
  params.cameras.resize(numCams);

  for (int i = 0; i < numCams; ++i)
  {
    params.cameras[i].P = cv::Mat_<float>(pMatrices.at(i));
  }

  // decompose projection matrices into K, R and t
  vector<Mat_<float> > K(numCams);
  vector<Mat_<float> > R(numCams);
  vector<Mat_<float> > T(numCams);

  vector<Mat_<float> > C(numCams);
  vector<Mat_<float> > t(numCams);

  for (size_t i = 0; i < numCams; i++)
  {
    cv::decomposeProjectionMatrix(params.cameras[i].P, K[i], R[i], T[i]);

    // get 3-dimensional translation vectors and camera center (divide by augmented component)
    C[i] = T[i](Range(0, 3), Range(0, 1)) / T[i](3, 0);
    t[i] = -R[i] * C[i];
  }
  Mat_<float> transform = Mat::eye(4, 4, CV_32F);
  params.cameras[0].reference = true;
  params.idRef = 0;

  //assuming K is the same for all cameras
  params.K = scaleK(K[0], scaleFactor);
  params.K_inv = params.K.inv();
  // get focal length from calibration matrix
  params.f = params.K(0, 0);

  for (size_t i = 0; i < numCams; i++)
  {
    params.cameras[i].R_orig_inv = R[i].inv(DECOMP_SVD);
    params.cameras[i].K = scaleK(K[i], scaleFactor);
    params.cameras[i].K_inv = params.cameras[i].K.inv();

    transformCamera(R[i], t[i], transform, params.cameras[i], params.K);

    params.cameras[i].P_inv = params.cameras[i].P.inv(DECOMP_SVD);
    params.cameras[i].M_inv = params.cameras[i].P.colRange(0, 3).inv();

    // set camera baseline (if unknown we need to guess something)
    params.cameras[i].baseline = 0.54f; //0.54 = Kitti baseline

    // K
    copyOpencvMatToFloatArray(params.cameras[i].K, &cpc.cameras[i].K);
    copyOpencvMatToFloatArray(params.cameras[i].K_inv, &cpc.cameras[i].K_inv);
    copyOpencvMatToFloatArray(params.cameras[i].R_orig_inv, &cpc.cameras[i].R_orig_inv);

    cpc.f = params.K(0, 0);
    cpc.cameras[i].f = params.K(0, 0);
    cpc.cameras[i].fx = params.K(0, 0);
    cpc.cameras[i].fy = params.K(1, 1);
    cpc.cameras[i].baseline = params.cameras[i].baseline;
    cpc.cameras[i].reference = params.cameras[i].reference;
    cpc.cameras[i].alpha = params.K(0, 0) / params.K(1, 1);

    // Copy data to cuda structure
    copyOpencvMatToFloatArray(params.cameras[i].P, &cpc.cameras[i].P);
    copyOpencvMatToFloatArray(params.cameras[i].P_inv, &cpc.cameras[i].P_inv);
    copyOpencvMatToFloatArray(params.cameras[i].M_inv, &cpc.cameras[i].M_inv);
    copyOpencvMatToFloatArray(params.cameras[i].K, &cpc.cameras[i].K);
    copyOpencvMatToFloatArray(params.cameras[i].K_inv, &cpc.cameras[i].K_inv);
    copyOpencvMatToFloatArray(params.cameras[i].R, &cpc.cameras[i].R);
    copyOpencvVecToFloat4(params.cameras[i].C, &cpc.cameras[i].C4);

    cpc.cameras[i].t4.x = params.cameras[i].t(0);
    cpc.cameras[i].t4.y = params.cameras[i].t(1);
    cpc.cameras[i].t4.z = params.cameras[i].t(2);

    Mat_<float> tmp = params.cameras[i].P.col(3);
    cpc.cameras[i].P_col34.x = tmp(0, 0);
    cpc.cameras[i].P_col34.y = tmp(1, 0);
    cpc.cameras[i].P_col34.z = tmp(2, 0);

    Mat_<float> tmpKinv = params.K_inv.t();
  }

  return params;
}

static int runGipuma(std::vector<cv::Mat> images,
                     std::vector<cv::Matx34f> pMatrices,
                     cv::Mat_<Vec3f> &generatedNormalMap,
                     cv::Mat_<float> &generatedDepthmap,
                     AlgorithmParameters &algParams,
                     GTcheckParameters &gtParameters
                    )
{
  // create folder to store result images
  time_t timeObj;
  time(&timeObj);
  tm *pTime = localtime(&timeObj);

  algParams.max_views = images.size();
  algParams.num_img_processed = images.size();
  size_t numImages = images.size();

  vector<Mat_<Vec3b> > img_color(numImages);
  for (size_t i = 0; i < numImages; i++)
  {
    img_color[i] = images[i];
  }

  uint32_t rows = img_color[0].rows;
  uint32_t cols = img_color[0].cols;
  uint32_t numPixels = rows * cols;

  size_t avail;
  size_t used;
  size_t total;

  GlobalState *gs = new GlobalState;

//  std::vector<cv::Matx34f> pMatrices;
//  for (auto element : images)
//  {
//    pMatrices.push_back(element->getPMatrix());
//  }

  CameraParameters cameraParams = modifiedGetCameraParameters(*(gs->cameras), pMatrices, algParams.cam_scale);

  //allocation for disparity and normal stores
  vector<Mat_<float> > disp(algParams.num_img_processed);
  vector<Mat_<uchar> > validCost(algParams.num_img_processed);
  for (int i = 0; i < algParams.num_img_processed; i++)
  {
    disp[i] = Mat::zeros(rows, cols, CV_32F);
    validCost[i] = Mat::zeros(rows, cols, CV_8U);
  }

  modifiedSelectViews(cameraParams, cols, rows, algParams);

  int numSelViews = images.size();

  cout << "Total number of images used: " << numSelViews << endl;
  cout << "Selected views: ";
  for (int i = 0; i < numSelViews; i++)
  {
    cout << cameraParams.viewSelectionSubset[i] << ", ";
    gs->cameras->viewSelectionSubset[i] = cameraParams.viewSelectionSubset[i];
  }
  cout << endl;

  for (int i = 0; i < algParams.num_img_processed; i++)
  {
    cameraParams.cameras[i].depthMin = algParams.depthMin;
    cameraParams.cameras[i].depthMax = algParams.depthMax;

    gs->cameras->cameras[i].depthMin = algParams.depthMin;
    gs->cameras->cameras[i].depthMax = algParams.depthMax;

    algParams.min_disparity =
        disparityDepthConversion(cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMax);
    algParams.max_disparity =
        disparityDepthConversion(cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMin);

    double minVal, maxVal;
    minMaxLoc(disp[i], &minVal, &maxVal);
  }
  cout << "Range of Minimum/Maximum depth is: " << algParams.min_disparity << " " << algParams.max_disparity
       << ", change it with --depth_min=<value> and  --depth_max=<value>" << endl;

  // run gpu run
  // Init parameters
  gs->params = &algParams;

  gs->cameras->viewSelectionSubsetNumber = numSelViews;

  // Init ImageInfo
  gs->cameras->cols = cols;
  gs->cameras->rows = rows;
  gs->params->cols = cols;
  gs->params->rows = rows;

  // Resize lines
  {
    gs->lines->n = rows * cols;
    gs->lines->resize(rows * cols);
    gs->lines->s = cols;
    gs->lines->l = cols;
  }

  vector<Mat> img_grayscale_float(numImages);
  vector<Mat> img_color_float(numImages);
  vector<Mat> img_color_float_alpha(numImages);
  vector<Mat_<uint16_t> > img_grayscale_uint(numImages);
  for (size_t i = 0; i < numImages; i++)
  {

    vector<Mat_<float> > rgbChannels(3);
    img_color_float_alpha[i] = Mat::zeros(rows, cols, CV_32FC4);
    img_color[i].convertTo(img_color_float[i], CV_32FC3); // or CV_32F works (too)
    Mat alpha(rows, cols, CV_32FC1);

    // add alpha channel
    split(img_color_float[i], rgbChannels);
    rgbChannels.push_back(alpha);
    merge(rgbChannels, img_color_float_alpha[i]);

  }
  int64_t t = getTickCount();

  cudaMemGetInfo(&avail, &total);
  cout << "Used memory before copy: " << (total - avail) / 1000000 << endl;
  // Copy images to texture memory
  addImageToTextureFloatColor(img_color_float_alpha, gs->imgs, gs->cuArray);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemGetInfo(&avail, &total);
  cout << "Used memory after copy: " << (total - avail) / 1000000 << endl;
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  runcuda(*gs);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  generatedNormalMap = Mat::zeros(img_color_float_alpha[0].rows, img_color_float_alpha[0].cols, CV_32FC3);
  generatedDepthmap = Mat::zeros(img_color_float_alpha[0].rows, img_color_float_alpha[0].cols, CV_32FC1);

  // Retreive the disparity image from cuda memory
  for (int i = 0; i < img_color_float_alpha[0].cols; i++)
  {
    for (int j = 0; j < img_color_float_alpha[0].rows; j++)
    {
      int center = i + img_color_float_alpha[0].cols * j;
      float4 n = gs->lines->norm4[center];
      generatedNormalMap(j, i) = Vec3f(n.x, n.y, n.z);
      generatedDepthmap(j, i) = gs->lines->norm4[i + img_color_float_alpha[0].cols * j].w;
    }
  }

  t = getTickCount() - t;
  double rt = (double) t / getTickFrequency();

  cout << "Total runtime including disk i/o: " << rt << "sec" << endl;

  // Free memory
  delTexture(algParams.num_img_processed, gs->imgs, gs->cuArray);
  delete gs;
  delete &algParams;
  cudaDeviceSynchronize();

  return 0;
}
#endif //VO3DRP_MODIFIEDGIPUMA_HPP
