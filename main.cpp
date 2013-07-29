/**
 * @file:   main.cpp
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 * @date:   Created on 2. Dezember 2012
 * @brief:  Example program on how to train your custom HOG detecting vector
 * for use with openCV <code>hog.setSVMDetector(_descriptor)</code>;
 * 
 * For the paper regarding Histograms of Oriented Gradients (HOG), @see http://lear.inrialpes.fr/pubs/2005/DT05/
 * You can populate the positive samples dir with files from the INRIA person detection dataset, @see http://pascal.inrialpes.fr/data/human/
 * This program uses SVMlight as machine learning algorithm (@see http://svmlight.joachims.org/), but is not restricted to it
 * Tested in Ubuntu Linux 64bit 12.04 "Precise Pangolin" with openCV 2.3.1, SVMlight 6.02, g++ 4.6.3
 * and standard HOG settings, training images of size 64x128px.
 * 
 * What this program basically does:
 * 1. Read positive and negative training sample image files from specified directories
 * 2. Calculate their HOG features and keep track of their classes (pos, neg)
 * 3. Save the feature map (vector of vectors/matrix) to file system
 * 4. Read in and pass the features and their classes to a machine learning algorithm, e.g. SVMlight
 * 5. Train the machine learning algorithm using the specified parameters
 * 6. Use the calculated support vectors and SVM model to calculate a single detecting descriptor vector
 * 
 * Build by issuing:
 * g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main.o.d -o main.o main.cpp
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
 * g++ `pkg-config --cflags opencv` -o trainhog main.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
 * 
 * Warning:
 * Be aware that the program may consume a considerable amount of main memory, hard disk memory and time, dependent on the amount of training samples.
 * Also be aware that (esp. for 32bit systems), there are limitations for the maximum file size which may take effect when writing the features file.
 * 
 * Terms of use:
 * This program is to be used as an example and is provided on an "as-is" basis without any warranties of any kind, either express or implied.
 * Use at your own risk.
 * For used third-party software, refer to their respective terms of use and licensing.
 */

#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "svmlight/svmlight.h"

using namespace std;
using namespace cv;

// <editor-fold defaultstate="collapsed" desc="Parameter definitions">
/* Parameter definitions */

// Directory containing positive sample images
static string posSamplesDir = "pos/";
// Directory containing negative sample images
static string negSamplesDir = "neg/";
// Set the file to write the features to
static string featuresFile = "genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "genfiles/svmlightmodel.dat";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorFile = "genfiles/descriptor.xml";
// Set the height of the HOG descriptor
static int HOGheight = 64;
// Set the width of the HOG descriptor
static int HOGwidth = 128;
// Set scale to > 1 to search for objects larger than HOGdescriptor
// for multiscaledetect searches will be done at a max of 64 scales by default
static double scale = 1.1;

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="Helper functions">
/* Helper functions */

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

/**
 * This method is unnecessary. OpenCV has an undocumented HOGDescriptor::write and
 * HOGDescriptor::load that allows writing and loading the entire HOGDescriptor.
 * Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
// static void saveDescriptorVectorToFile( vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
// }

/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 * @return 
 */
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

/**
 * I found this method to be unnecessary since it just calls hog.compute() 
 * after I made it possible to input any size images and split up negative images DMC
 * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
 * @param imageData cv::Mat to which feature will be calculated
 * @param descriptorVector the returned calculated feature vector<float> , 
 *      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
 * @param hog HOGDescriptor containin HOG settings
 */
// static void calculateFeaturesFromInput(const Mat imageData, vector<float>& featureVector, HOGDescriptor& hog) {
// }

/**
 * Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */
static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}

/**
 * Test detection with custom HOG description vector
 * @param hog
 * @param imageData
 */
static void detectTest(const HOGDescriptor& hog, Mat& imageData) {
    vector<Rect> found;
    int groupThreshold = 2;
    Size padding(Size(32, 32));
    Size winStride(Size(8, 8)); //must be multiple of (8,8)
    double hitThreshold = 0.; // tolerance
    //printf("hog levels = %d\n", hog.nlevels); //default is 64
    
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, scale, groupThreshold);
    showDetections(found, imageData);
}
// </editor-fold>

/**
 * Main program entry point
 * @param argc
 * @param argv
 * @return EXIT_SUCCESS (0) or EXIT_FAILURE (1)
 */
int main(int argc, char** argv) {

    // <editor-fold defaultstate="collapsed" desc="Init">
    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = Size(HOGwidth, HOGheight); // Chose this as average ship size
    // Get the files to train from somewhere
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Read image files">
    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    /// Retrieve the descriptor vectors from the samples
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Calculate HOG features and save to file">
    // Make sure there are actually samples to train
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }

    /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;
    /**
     * Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
     * @NOTE: If you split these steps into separate steps: 
     * 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >), 
     * 2. saving features to file / directly inject from memory to machine learning algorithm,
     * the program may consume a considerable amount of main memory
     */ 
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
		// Remove following line for libsvm which does not support comments
        // File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
        // Iterate over sample images
        int numpos = positiveTrainingImages.size();
        int featurelength;
        Mat im;
        Mat imageData = Mat(hog.winSize, 0);
        int fwidth = hog.winSize.width;
		int fheight = hog.winSize.height;
		int pointr, pointc = 0;
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            // Get positive or negative sample image file path
            const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
			
			im = imread(currentImageFile, 0);
    		if (im.empty()) {
        		featureVector.clear();
        		printf("Error: HOG image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
    		} 
    		else { // Image exists
    		
				// Calculate feature vector from current image file
				if (currentFile < numpos) { //shink positive images to feature size
					resize(im, imageData, hog.winSize, NULL, NULL, INTER_LINEAR);
					hog.compute(imageData, featureVector, winStride, trainingPadding);
					if (!featureVector.empty()) {
						featurelength=featureVector.size();
						/* Put positive or negative sample class to file, 
						 * true=positive, false=negative, 
						 * and convert positive class to +1 and negative class to -1 for SVMlight
						 */
						//File << ((currentFile < numpos) ? "+1" : "-1");
						File << "+1";
						// Save feature vector components
						for (unsigned int feature = 0; feature < featurelength; ++feature) {
							File << " " << (feature + 1) << ":" << featureVector.at(feature);
						}
						File << endl;
					}
				}
				else { //Chop up negative images into blocks that are the size of the feature
					//loop through columns
					for(pointc = 0; pointc + fwidth < im.cols; pointc+=fwidth/2){
					//loop through rows
						for(pointr = 0; pointr + fheight < im.rows; pointr+=fwidth/2){
							imageData = im.colRange(pointc,pointc+fwidth).rowRange(pointr,pointr+fheight);
							hog.compute(imageData, featureVector, winStride, trainingPadding);
							if (!featureVector.empty()) {
								/* Put positive or negative sample class to file, 
								 * true=positive, false=negative, 
								 * and convert positive class to +1 and negative class to -1 for SVMlight
								 */
								//File << ((currentFile < numpos) ? "+1" : "-1");
								for ( unsigned int negsamples = 0; negsamples < featureVector.size(); negsamples += featurelength){
									File << "-1";
									// Save feature vector components
									for (unsigned int feature = negsamples; feature < negsamples + featurelength; ++feature) {
										File << " " << (feature + 1) << ":" << featureVector.at(feature);
									}
									File << endl;
								}
							}
						}
				
        			}
        		}
        		im.release();
        	}
     	}
     	imageData.release();
        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Pass features to machine learning algorithm">
    /// Read in and train the calculated feature vectors
    printf("Calling SVMlight\n");
    SVMlight::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
    SVMlight::getInstance()->train(); // Call the core libsvm training procedure
    printf("Training done, saving model file!\n");
    SVMlight::getInstance()->saveModelToFile(svmModelFile);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Generate single detecting feature vector from calculated SVM support vectors and SVM model">
    printf("Generating representative single HOG feature vector using svmlight!\n");
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    // Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
    SVMlight::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);

    // </editor-fold>

	hog.setSVMDetector(descriptorVector); // Set our custom detecting vector

    // And save the precious to file system
    FileStorage fs(descriptorFile, FileStorage::WRITE);
    string objName = "descriptor";  // Must name the obj being saved in the xml
	hog.write(fs,objName);
	fs.release();
	

	HOGDescriptor hogLoad; // Load the newly created hog descriptor
	string loadName = "descriptor";  // In order to load the descriptor objName must match the saved name
	hogLoad.load(descriptorFile,loadName);

	
	//  Uncomment to test postest.jpg stored in working directory
    printf("Testing on image\n");
	Mat testImage = imread("postest.jpg");
    if (testImage.empty()) {
        printf("Error: HOG test image is empty, features calculation skipped!\n");
        return EXIT_FAILURE
        ;
    }
	
    detectTest(hogLoad, testImage);
    imshow("HOG custom detection", testImage);
    waitKey();

	//  Uncomment to test negtest.jpg stored in working directory
// 	Mat testImageNeg = imread("negtest.jpg");
//     if (testImageNeg.empty()) {
//         printf("Error: HOG Neg test image is empty, features calculation skipped!\n");
//         return EXIT_FAILURE
//         ;
//     }
// 
//     detectTest(hogLoad, testImageNeg);
//     imshow("HOG custom detection", testImageNeg);
//     waitKey();

    return EXIT_SUCCESS;
}
