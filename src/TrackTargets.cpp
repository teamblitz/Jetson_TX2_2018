#include <cscore.h>
#include <ntcore.h>
#include <networktables/NetworkTable.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

#include "GripPipeline.h"

using namespace std;
using namespace cv;
using namespace grip;
using namespace llvm;

// Code conditionals.                   // Defaults
#define USE_CAMERA_INPUT            1   // 1   0 == Use test video from a file below, 1 == Use an attached video camera.
#define CAMERA_PORT                 1   // 1   0 == Jetson TX2 On-Board Camera, 1 == USB Cameraa.
#define RESTREAM_VIDEO              0   // 1
#define USE_CONTOUR_DETECTION       1   // 1   0 == Simple blob detection, 1 == Contour detection.
#define VIEW_OUTPUT_ON_DISPLAY      1   // 0
#define NON_ROBOT_NETWORK_TABLES    1   // 0
#define OUTLINE_VIEWER_IP_ADDRESS	"10.100.196.89"
#define DUMP_OPENCV_BUILD_INFO      0   // 0   1 == Output OpenCV build info when started.
#define MEASURE_PERFORMANCE         1   // 0   1 == Output timing measurements when running.

// Performance macros.
map<string, int64> ticks;
#if MEASURE_PERFORMANCE
#define TICK_ACCUMULATOR_START(NAME)    auto NAME ## Start = getTickCount()
#define TICK_ACCUMULATOR_END(NAME)      ticks[#NAME] += (getTickCount() - NAME ## Start)
#else
#define TICK_ACCUMULATOR_START(NAME)
#define TICK_ACCUMULATOR_END(NAME)
#endif

// Constants.
static const float FRAME_SCALE_FACTOR = 1.0;

// Forward declarations.
shared_ptr<NetworkTable> initializeNetworkTables();

void runContourDetectionPipeline(Mat const& frame,
                                 vector<vector<Point>>& hits,
                                 vector<Rect>& hitRects,
                                 vector<vector<Point>>& skips);
void runBlobDetectionPipeline(Ptr<SimpleBlobDetector> const& detector,
                              Mat const& frame,
                              vector<vector<Point>>& hits,
                              vector<Rect>& hitRects,
                              vector<vector<Point>>& skips);

void processKeyPoints(vector<KeyPoint> const& keypoints,
                      vector<vector<Point>>& hits,
                      vector<Rect>& hitRects,
                      vector<vector<Point>>& skips);
void processContours(vector<vector<Point>> const& contours,
                     vector<vector<Point>>& hits,
                     vector<Rect>& hitRects,
                     vector<vector<Point>>& skips);

SimpleBlobDetector::Params getSimpleBlobDetectorParams();
void hslThreshold(Mat const& input,
                  double hue[],
                  double sat[],
                  double lum[],
                  Mat& out);
void cvDilate(Mat const& src,
              Mat &kernel,
              Point &anchor,
              double iterations,
              int borderType,
              Scalar &borderValue,
              Mat& dst);
void findContours(Mat const& input,
                  bool externalOnly,
                  vector<vector<Point>>& contours);
void filterContours(vector<vector<Point>> const& inputContours,
                    double minArea,
                    double minPerimeter,
                    double minWidth, double maxWidth,
                    double minHeight, double maxHeight,
                    double solidity[],
                    double maxVertexCount, double minVertexCount,
                    double minRatio, double maxRatio,
                    vector<vector<Point>>& output);
void keyPointToPointsAndRect(KeyPoint const& keyPoint,
                             vector<Point>& points,
                             Rect& rect);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

int main(__attribute__((unused)) int argc, char** argv)
{
#if DUMP_OPENCV_BUILD_INFO
	// Dump OpenCV build info.
	std::cout << cv::getBuildInformation() << std::endl;
#endif

    cout << argv[0] << " running..." << endl;
#if MEASURE_PERFORMANCE
    auto startTicks = getTickCount();
    int64 frameCount = 0;
#endif

    auto ttTable = initializeNetworkTables();

#if RESTREAM_VIDEO
    // Create an MJPEG server for restreaming the USB camera feed
    // to the roboRIO.
    cs::CvSource restreamSource("CV Image Source", cs::VideoMode::PixelFormat::kMJPEG, 640, 480, 30);
    cs::MjpegServer mjpegServer("Image Server", 1186);
    mjpegServer.SetSource(restreamSource);
#endif

#if USE_CAMERA_INPUT
    // Open USB camera.
#if CAMERA_PORT == 0
	VideoCapture input("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
#else
    VideoCapture input(CAMERA_PORT);
#endif
    
	if (!input.isOpened())
    {
        cerr << "ERROR: Failed to open camera!" << endl;
        cout << "Make sure that there are no other instances of this program already running!" << endl;
        return -1;
    }
#else
    // Open a test video file.
//    VideoCapture input("../sample_media/videos/WIN_20170307_20_43_09_Pro.mp4");
    VideoCapture input("../sample_media/videos/WIN_20170307_20_45_18_Pro.mp4");
//    VideoCapture input("../sample_media/videos/WIN_20170314_19_22_47_Pro.mp4");
//    VideoCapture input("../sample_media/videos/WIN_20170314_19_24_21_Pro.mp4");
//    VideoCapture input("../sample_media/videos/WIN_20170314_19_25_35_Pro.mp4");

    if (!input.isOpened())
    {
        cout << "Could not open test video file. Reverting to live camera feed." << endl;
        input.open(0);
        if (!input.isOpened())
        {
            cerr << "ERROR: Failed to open camera!" << endl;
            cout << "Make sure that there are no other instances of this program already running!" << endl;
            return -1;
        }
    }
#endif

    SimpleBlobDetector::Params params = getSimpleBlobDetectorParams();
    Ptr<SimpleBlobDetector> blobDetector = SimpleBlobDetector::create(params);

    // Analytics.
    double minArea = 1000000;
    double maxArea = 0;

    // Grab and process frames.
    for (;;)
    {
        TICK_ACCUMULATOR_START(read);
        Mat rawFrame;
        if (!input.read(rawFrame))
            continue;
        TICK_ACCUMULATOR_END(read);
#if MEASURE_PERFORMANCE
        frameCount++;
#endif

        TICK_ACCUMULATOR_START(resize);
        Mat frame;
        resize(rawFrame, frame, Size(), FRAME_SCALE_FACTOR, FRAME_SCALE_FACTOR, CV_INTER_AREA);
        TICK_ACCUMULATOR_END(resize);

        vector<vector<Point>> hits, skips;
        vector<Rect> hitRects;

#if USE_CONTOUR_DETECTION
        runContourDetectionPipeline(frame, hits, hitRects, skips);
#else
        runBlobDetectionPipeline(blobDetector, frame, hits, hitRects, skips);
#endif

        TICK_ACCUMULATOR_START(network_tables);
        Point displayCenter;
        Rect displayRect;
        if (hits.size() == 1)
        {
            cout << "Hit" << endl;

            // Compute the bounding rect of the target.
            vector<double> center(2, 0.0);
            vector<double> rect(4, 0.0);
            auto left = hitRects[0].x;
            auto top = hitRects[0].y;
            auto right = hitRects[0].x + hitRects[0].width;
            auto bottom = hitRects[0].y + hitRects[0].height;
            auto centerX = (left + right)/2;
            auto centerY = (top + bottom)/2;

            // Compute a matching display rect.
            displayCenter.x = centerX;
            displayCenter.y = centerY;
            displayRect = hitRects[0];
            double area = displayRect.width * displayRect.height;
            if (area < minArea)
            {
                minArea = area;
            }
            if (area > maxArea)
            {
                maxArea = area;
            }

            // Send target info to network tables.
            center[0] = centerX;
            center[1] = centerY;
            rect[0] = top;                    // top, left, bottom, right
            rect[1] = left;
            rect[2] = bottom;
            rect[3] = right;

            ArrayRef<double> centerArray(center);
            ArrayRef<double> rectArray(rect);
            ttTable->PutBoolean("tracking", true);
            ttTable->PutNumberArray("center", centerArray);
            ttTable->PutNumberArray("rect", rectArray);
            ttTable->PutNumber("area", area);
        }
        else
        {
            cout << "Miss" << endl;

            ArrayRef<double> centerArray {0, 0, 0, 0};
            ArrayRef<double> rectArray {0, 0, 0, 0};
            ttTable->PutBoolean("tracking", false);
            ttTable->PutNumberArray("center", centerArray);
            ttTable->PutNumberArray("rect", rectArray);
            ttTable->PutNumber("area", 0);
        }
        TICK_ACCUMULATOR_END(network_tables);

        // Render the keypoints onto the frames.
        TICK_ACCUMULATOR_START(view);
#if RESTREAM_VIDEO || VIEW_OUTPUT_ON_DISPLAY
        Mat detectionFrame;
        frame.copyTo(detectionFrame);
        rectangle(detectionFrame, displayRect, Scalar(0, 255, 0), 4);	// green
        circle(detectionFrame, displayCenter, 4, Scalar(0, 255, 0), 2);	// green
        drawContours(detectionFrame, hits, -1, Scalar(0, 0, 255), 2);	// red
        drawContours(detectionFrame, skips, -1, Scalar(0, 0, 0), 2);	// black
#endif

#if RESTREAM_VIDEO
        restreamSource.PutFrame(detectionFrame);
#endif

#if VIEW_OUTPUT_ON_DISPLAY
        imshow("frame", detectionFrame);
        waitKey(1);
#endif
        TICK_ACCUMULATOR_END(view);
    }

#if MEASURE_PERFORMANCE
    const auto finalTicks = getTickCount();
    const auto tickFreq = getTickFrequency();
    const auto totalTicks = finalTicks - startTicks;
    const auto totalTime = totalTicks/getTickFrequency();    // seconds
    auto other = totalTicks;
    cout << "Execution total time: " << totalTime << " seconds" << endl;
    for (auto const& entry : ticks)
    {
        cout << "  " << entry.first << ": " << entry.second/tickFreq << " seconds" << endl;
        other -= entry.second;
    }
    cout << "  other: " << other/tickFreq << " seconds" << endl;
    cout << "Frames processed: " << frameCount << endl;
    cout << "Frame rate: " << frameCount/totalTime << " frames/second" << endl;
    cout << "Min Area: " << minArea << endl;
    cout << "Max Area: " << maxArea << endl;
#endif

    // Clean up and shutdown.
    input.release();
    cout << argv[0] << " finished!" << endl;
}

shared_ptr<NetworkTable> initializeNetworkTables()
{
    // Connect NetworkTables and get access to the tracking table.
    NetworkTable::SetClientMode();
    NetworkTable::SetTeam(2083);
    
#if NON_ROBOT_NETWORK_TABLES
    // Change this address to the dynamically-generated
    // TCP/IP address of the computer (not roboRIO) that
    // is running a NetworkTables intance in server mode.
    NetworkTable::SetIPAddress(OUTLINE_VIEWER_IP_ADDRESS);
#endif

    NetworkTable::Initialize();

    return NetworkTable::GetTable("target_tracking");
}

void runContourDetectionPipeline(Mat const& frame,
                                 vector<vector<Point>>& hits,
                                 vector<Rect>& hitRects,
                                 vector<vector<Point>>& skips)
{
    TICK_ACCUMULATOR_START(hsl_threshold);
    Mat hslFrame;
	double hslThresholdHue[] = {55.0, 115.0};
	double hslThresholdSaturation[] = {175.0, 255.0};
	double hslThresholdLuminance[] = {50.0, 255.0};
    hslThreshold(frame, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, hslFrame);
    TICK_ACCUMULATOR_END(hsl_threshold);

    TICK_ACCUMULATOR_START(dilation);
    Mat dilationFrame;
	Mat cvDilateKernel;
	Point cvDilateAnchor(-1, -1);
	double cvDilateIterations = 1.0;                // 1.0
    int cvDilateBordertype = BORDER_CONSTANT;       // BORDER_CONSTANT
	Scalar cvDilateBordervalue(-1);                 // -1
	cvDilate(hslFrame, cvDilateKernel, cvDilateAnchor,
             cvDilateIterations, cvDilateBordertype,
             cvDilateBordervalue, dilationFrame);
    TICK_ACCUMULATOR_END(dilation);

    TICK_ACCUMULATOR_START(find_contours);
    vector<vector<Point>> foundContours;
	bool findContoursExternalOnly = false;          // false
	findContours(dilationFrame, findContoursExternalOnly, foundContours);
    TICK_ACCUMULATOR_END(find_contours);

    TICK_ACCUMULATOR_START(filtered_contours);
    vector<vector<Point>> filteredContours;
	double filterContoursMinArea = 150.0; 
	double filterContoursMinPerimeter = 0;          // 0.0
	double filterContoursMinWidth = 0.0;            // 0.0
	double filterContoursMaxWidth = 20000.0;         // 1000.0
	double filterContoursMinHeight = 0.0;           // 0.0
	double filterContoursMaxHeight = 20000.;         // 1000.0
	double filterContoursSolidity[] = {0.0, 100};
	double filterContoursMaxVertices = 1000000;     // 1000000.0
	double filterContoursMinVertices = 0;           // 0.0
	double filterContoursMinRatio = 0.0;            // 0.0
	double filterContoursMaxRatio = 1000.0;         // 1000.0
	filterContours(foundContours,
                   filterContoursMinArea,
                   filterContoursMinPerimeter,
                   filterContoursMinWidth,
                   filterContoursMaxWidth,
                   filterContoursMinHeight,
                   filterContoursMaxHeight,
                   filterContoursSolidity,
                   filterContoursMaxVertices,
                   filterContoursMinVertices,
                   filterContoursMinRatio,
                   filterContoursMaxRatio,
                   filteredContours);
    TICK_ACCUMULATOR_END(filtered_contours);

    TICK_ACCUMULATOR_START(filter);
    processContours(filteredContours, hits, hitRects, skips);
    TICK_ACCUMULATOR_END(filter);
}

void runBlobDetectionPipeline(Ptr<SimpleBlobDetector> const& detector,
                              Mat const& frame,
                              vector<vector<Point>>& hits,
                              vector<Rect>& hitRects,
                              vector<vector<Point>>& skips)
{
//    TICK_ACCUMULATOR_START(blur);
//    Mat blurredFrame;
//    medianBlur(frame, blurredFrame, 11);
//    TICK_ACCUMULATOR_END(blur);
    
//     TICK_ACCUMULATOR_START(threshold);
//     Mat thresholdFrame;
//     threshold(frame, thresholdFrame, 200, 255, CV_THRESH_BINARY);
//     TICK_ACCUMULATOR_END(threshold);

    TICK_ACCUMULATOR_START(detect);
    vector<KeyPoint> keyPoints;
    detector->detect(frame, keyPoints); 
    TICK_ACCUMULATOR_END(detect);

    TICK_ACCUMULATOR_START(sort);
    sort(keyPoints.begin(), keyPoints.end(),
         [] (KeyPoint const& a, KeyPoint const& b) { return a.size > b.size; });
    TICK_ACCUMULATOR_END(sort);

    TICK_ACCUMULATOR_START(filter);
    processKeyPoints(keyPoints, hits, hitRects, skips);
    TICK_ACCUMULATOR_END(filter);
}

/**
 * Filter the keypoints looking for potential keypoints that
 * correspond to the closest (largest) piece of retrorefledtive
 * tape. Once found, the target keypoints will be returned
 * in the "hits" vector. Any keypoints skipped will be
 * returned in the "skips" vector.
 */
void processKeyPoints(vector<KeyPoint> const& keyPoints,
                      vector<vector<Point>>& hits,
                      vector<Rect>& hitRects,
                      vector<vector<Point>>& skips)
{
    if (keyPoints.size() == 0)
    {
    	// No keypoints found in frame.
        return;
    }

	// One or more keypoints was detected and, since they are sorted,
	// return the data for the first (largest) keypoint as a hit.
	vector<Point> hitPoints;
	Rect hitRect;
	keyPointToPointsAndRect(*(keyPoints.begin()), hitPoints, hitRect);
	hits.push_back(hitPoints);
	hitRects.push_back(hitRect);

	// Return everything else as a skip.
	if (keyPoints.size() > 1)
    {	
		for (auto iter = keyPoints.begin()+1; iter != keyPoints.end(); iter++)
		{    
			vector<Point> skipPoints;
			Rect skipRect;
			keyPointToPointsAndRect(*iter, skipPoints, skipRect);
			skips.push_back(skipPoints);
		}
    }
}

/**
 * Filter the contours...
 */
void processContours(vector<vector<Point>> const& contours,
                     vector<vector<Point>>& hits,
                     vector<Rect>& hitRects,
                     vector<vector<Point>>& skips)
{
    if (contours.size() == 0)
    {
    	// No contours found in frame.
        return;
    }

	// One or more contours was detected. Find the best candidate for a hit
	// and return the rest as skips.
    for (auto iter = contours.begin(); iter != contours.end(); iter++)
    {
        Rect rect = boundingRect(*iter);
        if (rect.height < 5 * rect.width || rect.height > 10 * rect.width)
        {
//            cout << "Skipping contour (aspect ratio)..." << endl;
            continue;
        }

        if (rect.area() > 20000)
        {
//            cout << "Skipping contour (area)..." << rect.area() << endl;
            continue;
        }

        for (auto other = iter+1; other != contours.end(); other++)
        {
            Rect rectOther = boundingRect(Mat(*other));
            if (rectOther.height < 1 * rectOther.width || rectOther.height > 4 * rectOther.width)
            {
//                cout << "Skipping contour (aspect ratio)..." << endl;
                continue;
            }

            if (rectOther.area() > 20000)
            {
//                cout << "Skipping contour (area)..." << rect.area() << endl;
                continue;
            }

            auto avgTargetWidth = (rect.width + rectOther.width)/2;
            auto avgTargetHeight = (rect.height + rectOther.height)/2;
//            cout << "Averages: " << avgTargetWidth << " " << avgTargetHeight << endl;

            auto rectX = rect.x + rect.width/2;
            auto rectOtherX = rectOther.x + rectOther.width/2;
//            cout << "x: " << rectX << " " << rectOtherX << endl;

            auto rectY = rect.y + rect.height/2;
            auto rectOtherY = rectOther.y + rectOther.height/2;
//            cout << "y: " << rectY << " " << rectOtherY << endl;

            auto deltaX = abs(rectX - rectOtherX);
            auto deltaY =  abs(rectY - rectOtherY);
//            cout << "Deltas: " << deltaX << " " << deltaY << endl;

            if ((deltaX < 7 * avgTargetWidth) && (deltaX > 2 * avgTargetWidth) &&
                (deltaY < 1.0 * avgTargetHeight))
            {
                hits.push_back(*iter);
                hits.push_back(*other);
                hitRects.push_back(rect);
                hitRects.push_back(rectOther);
                return;
            }
            else
            {
//                cout << "Skipping contour (filters)..." << endl;
                skips.push_back(*iter);
            }
        }   
    }
}

SimpleBlobDetector::Params getSimpleBlobDetectorParams()
{
    const float fsfs = FRAME_SCALE_FACTOR * FRAME_SCALE_FACTOR;

    SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
    params.thresholdStep = 10;				// 10
    params.minThreshold = 180;				// 180
    params.maxThreshold = 255;				// 255
    params.minRepeatability = 2;			// 2
    params.minDistBetweenBlobs = 10;		// 10
    params.filterByColor = false;			// false; possibly broken in OpenCV
    params.blobColor = 255;					// 255
    params.filterByArea = true;				// true
    params.minArea = 1000 * fsfs;			// 1000
    params.maxArea = 100000 * fsfs;			// INT_MAX
    params.filterByCircularity = false;		// false; 1 == circle, 0.785 == square, etc.
    params.minCircularity = 0.0;			// 0.0
    params.maxCircularity = 0.2;			// 0.2
    params.filterByInertia = true;			// true; 0 == line, 1 == circle
    params.minInertiaRatio = 0.0;			// 0.0
    params.maxInertiaRatio = 0.1;			// 0.1
    params.filterByConvexity = true;		// true; 0 == concave; 1 == convex
    params.minConvexity = 0.9;				// 0.9
    params.maxConvexity = 1.0;				// 1.0

    return params;
}

/**
 * Segment an image based on hue, saturation, and luminance ranges.
 *
 * @param input The image on which to perform the HSL threshold.
 * @param hue The min and max hue.
 * @param sat The min and max saturation.
 * @param lum The min and max luminance.
 * @param output The image in which to store the output.
 */
void hslThreshold(Mat const& input,
                  double hue[],
                  double sat[],
                  double lum[],
                  Mat& out)
{
	cvtColor(input, out, COLOR_BGR2HLS);
	inRange(out, Scalar(hue[0], lum[0], sat[0]), Scalar(hue[1], lum[1], sat[1]), out);
}

/**
 * Expands area of higher value in an image.
 * @param src the Image to dilate.
 * @param kernel the kernel for dilation.
 * @param anchor the center of the kernel.
 * @param iterations the number of times to perform the dilation.
 * @param borderType pixel extrapolation method.
 * @param borderValue value to be used for a constant border.
 * @param dst Output Image.
 */
void cvDilate(Mat const& src,
              Mat &kernel,
              Point &anchor,
              double iterations,
              int borderType,
              Scalar &borderValue,
              Mat& dst)
{
    dilate(src, dst, kernel, anchor, (int)iterations, borderType, borderValue);
}

/**
 * Finds contours in an image.
 *
 * @param input The image to find contours in.
 * @param externalOnly if only external contours are to be found.
 * @param contours vector of contours to put contours in.
 */
void findContours(Mat const& input, bool externalOnly, vector<vector<Point>>& contours)
{
	vector<Vec4i> hierarchy;
	contours.clear();
	int mode = externalOnly ? RETR_EXTERNAL : RETR_LIST;
	int method = CHAIN_APPROX_SIMPLE;
	findContours(input, contours, hierarchy, mode, method);
}

/**
 * Filters through contours.
 * @param inputContours is the input vector of contours.
 * @param minArea is the minimum area of a contour that will be kept.
 * @param minPerimeter is the minimum perimeter of a contour that will be kept.
 * @param minWidth minimum width of a contour.
 * @param maxWidth maximum width.
 * @param minHeight minimum height.
 * @param maxHeight  maximimum height.
 * @param solidity the minimum and maximum solidity of a contour.
 * @param minVertexCount minimum vertex Count of the contours.
 * @param maxVertexCount maximum vertex Count.
 * @param minRatio minimum ratio of width to height.
 * @param maxRatio maximum ratio of width to height.
 * @param output vector of filtered contours.
 */
void filterContours(vector<vector<Point>> const& inputContours,
                    double minArea,
                    double minPerimeter,
                    double minWidth, double maxWidth,
                    double minHeight, double maxHeight,
                    double solidity[],
                    double maxVertexCount, double minVertexCount,
                    double minRatio, double maxRatio,
                    vector<vector<Point>> &output)
{
    vector<Point> hull;
    output.clear();
    for (vector<Point> contour: inputContours)
    {
    	Rect bb = boundingRect(contour);
    	if (bb.width < minWidth || bb.width > maxWidth) continue;
    	if (bb.height < minHeight || bb.height > maxHeight) continue;
    	double area = contourArea(contour);
    	if (area < minArea) continue;
    	if (arcLength(contour, true) < minPerimeter) continue;
    	convexHull(Mat(contour, true), hull);
    	double solid = 100 * area / contourArea(hull);
    	if (solid < solidity[0] || solid > solidity[1]) continue;
    	if (contour.size() < minVertexCount || contour.size() > maxVertexCount)	continue;
    	double ratio = (double) bb.width / (double) bb.height;
    	if (ratio < minRatio || ratio > maxRatio) continue;
    	output.push_back(contour);
    }
}

void keyPointToPointsAndRect(KeyPoint const& keyPoint,
                             vector<Point>& points,
                             Rect& rect)
{
    // Convert keypoints to vector of points in the shape of a rectangle.
    float left = keyPoint.pt.x - keyPoint.size/2;
    float top = keyPoint.pt.y - keyPoint.size/2;
    float right = left + keyPoint.size;
    float bottom = top + keyPoint.size;
    points.push_back(Point(left, top));
    points.push_back(Point(right, top));
    points.push_back(Point(right, bottom));
    points.push_back(Point(left, bottom));
    points.push_back(Point(left, top));
    rect.x = left;
    rect.y = top;
    rect.width = keyPoint.size;
    rect.height = keyPoint.size;
}

#pragma GCC diagnostic pop

