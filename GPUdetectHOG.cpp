/*  
 *	This HOG detector requires OpenCV built with GPU support.
 *	The Program will display results of HOG detection for both CPU and GPU
 *	on the image_file provided.
 *  File:    detectHOG.cpp
 *  Author:  David Camp
 *  Created: Jul 29, 2013
 */
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

int main(int argc, char** argv) {

    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices == 0)
    {
        cout << "No GPU found or the library was compiled without GPU support";
    	return -1;
    }

    const char* keys =
       "{ h | help    | false | print help message }"
       "{ d | device  | 0     | GPU device id }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        cmd.printParams();
        printf("Usage: detectHOG HOGDescriptor_file image_file [--device=[x]]\n");
        return 0;
    }
    
    int device = cmd.get<int>("device");
    if (device < 0 || device >= num_devices)
    {
        cerr << "Invalid device ID" << endl;
        return -1;
    }
    DeviceInfo dev_info(device);
    if (!dev_info.isCompatible())
    {
        cerr << "GPU module isn't built for GPU #" << device << " " << dev_info.name() << ", CC " << dev_info.majorVersion() << '.' << dev_info.minorVersion() << endl;
        return -1;
    }
    
    // check arguments
	if (argc != 3 && argc != 4 ) {
        printf("Usage: detectHOG HOGDescriptor_file image_file [--device=[x]]\n");
		exit(-1);
	}
    
    setDevice(device);
    printShortCudaDeviceInfo(device);
    
    string DescriptorName = "HOGDescriptor";
    string DescriptorFile = argv[1];
    
    // load the image from file
	Mat im = imread(argv[2]);
	if (im.empty()) {
		printf("Image not found or invalid image format\n");
		exit(-1);
	}
	// load separate image files for displaying detections one for CPU one for GPU.
    Mat display = imread(argv[2]);
    Mat displaygpu = imread(argv[2]);
    
    // Create HOGDescriptor for CPU
    cv::HOGDescriptor hog;
    if(!hog.load(DescriptorFile,DescriptorName)){
		printf("Unable to load in HOGDescriptor.\n");
		exit(-1);
	}
    
    // Perform detection with CPU
	std::vector<cv::Rect> found;
	hog.detectMultiScale(im,found);
	
	// Display detection by CPU
	for(unsigned i = 0; i < found.size(); i++) {
			cv::Rect r = found[i];
			rectangle(display, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
		}
	cv::imshow("CPU Results", display);
 
 	// COPY HOGDescriptor to GPU.  I could not find a built in function to do this.
 	// Also note that even though all parameters are passed not all parameters are used by
 	// the construction.  gpu::HOGEDescriptor doesn't have the full functionality of 
 	// cv::HOGDescriptor
 	cv::gpu::HOGDescriptor GPUhog = cv::gpu::HOGDescriptor::HOGDescriptor(hog.winSize, hog.blockSize, hog.blockStride, hog.cellSize, hog.nbins, hog.winSigma, hog.L2HysThreshold, hog.gammaCorrection, hog.nlevels);
	GPUhog.setSVMDetector(hog.svmDetector);
    
    // Copy the image to the GPU
	GpuMat imgpu, gimgpu;
	std::vector<cv::Rect> gfound;
	imgpu.upload(im);
	// gpu::HOGDescriptor::detectMultiScale only can handle two GpuMat types and one is
	// Gray scale so converting to grayscale prevents error if file is wrong type.
	cvtColor(imgpu, gimgpu, CV_BGR2GRAY);
	
	// Run Detection on GPUG
	GPUhog.detectMultiScale(gimgpu,gfound);
	 
	// This waitKey() allow the GPU to process while observing the CPU image 
	waitKey();
	
	// Display detection by GPU
	for(unsigned i = 0; i < gfound.size(); i++) {
			cv::Rect gr = gfound[i];
			rectangle(displaygpu, gr.tl(), gr.br(), cv::Scalar(0,255,0), 2);
	}
	cv::imshow("GPU Results", displaygpu);
	waitKey();
	
	//Release
	im.release();
	display.release();
	displaygpu.release();
	imgpu.release();
	gimgpu.release();
}