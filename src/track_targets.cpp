#include "cscore.h"
#include "networktables/NetworkTable.h"
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

// Code conditionals.
#define USE_CAMERA_INPUT            0   // 1
#define RESTREAM_VIDEO              1   // 1
#define USE_CONTOUR_DETECTION       1   // 1   vs. 0 ==  simple blob detection
#define VIEW_OUTPUT_ON_DISPLAY      0   // 0
#define NON_ROBOT_NETWORK_TABLES    0   // 0
#define MEASURE_PERFORMANCE         1   // 0

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
static const float FRAME_SCALE_FACTOR = 0.5;

// Forward declarations.
shared_ptr<NetworkTable> initializeNetworkTables();

void runContourDetectionPipeline(Mat const& frame,
                                 vector<vector<Point>>& hits,
                                 vector<Rect>& hitRects,
                                 vector<vector<Point>>& skips);
void runBlobDetectionPipeline(SimpleBlobDetector const& detector,
                              Mat const& frame,
                              vector<vector<Point>>& hits,
                              vector<Rect>& hitRects,
                              vector<vector<Point>>& skips);

void filterKeyPoints(vector<KeyPoint> const& keypoints,
                     vector<vector<Point>>& hits,
                     vector<Rect>& hitRects,
                     vector<vector<Point>>& skips);
void filterContours(vector<vector<Point>> const& contours,
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

int main(int argc, char** argv)
{
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
    // Open USB camera on port 0.
    VideoCapture input(0);
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
    SimpleBlobDetector blobDetector(params);

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
        if (hits.size() > 1)
        {
            cout << "Hits" << endl;

            // Compute a rect that covers both targets.
            vector<double> center(2, 0.0);
            vector<double> rect(4, 0.0);
            auto left = min(hitRects[0].x, hitRects[1].x);
            auto top = min(hitRects[0].y, hitRects[1].y);
            auto right0 = hitRects[0].x + hitRects[0].width;
            auto right1 = hitRects[1].x + hitRects[0].width;
            auto right = max(right0, right1);
            auto bottom0 = hitRects[0].y + hitRects[0].height;
            auto bottom1 = hitRects[1].y + hitRects[0].height;
            auto bottom = max(bottom0, bottom1);
            auto centerX = (left + right)/2;
            auto centerY = (top + bottom)/2;

            // Compute the center of both targets combined.
            displayCenter.x = centerX;
            displayCenter.y = centerY;
            displayRect.x = left;
            displayRect.y = top;
            displayRect.width = abs(right - left);
            displayRect.height = abs(bottom - top);
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
        rectangle(detectionFrame, displayRect, Scalar(0, 255, 0), 2);
        circle(detectionFrame, displayCenter, 4, Scalar(0, 255, 0), 2);
        drawContours(detectionFrame, hits, -1, Scalar(0, 0, 255), 2);
        drawContours(detectionFrame, skips, -1, Scalar(0, 0, 0), 2);
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
    NetworkTable::SetIPAddress("169.254.25.155");
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
	double filterContoursMaxWidth = 1000.0;         // 1000.0
	double filterContoursMinHeight = 0.0;           // 0.0
	double filterContoursMaxHeight = 1000.;         // 1000.0
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
    filterContours(filteredContours, hits, hitRects, skips);
    TICK_ACCUMULATOR_END(filter);
}

void runBlobDetectionPipeline(SimpleBlobDetector const& detector,
                              Mat const& frame,
                              vector<vector<Point>>& hits,
                              vector<Rect>& hitRects,
                              vector<vector<Point>>& skips)
{
//    TICK_ACCUMULATOR_START(blur);
//    Mat blurredFrame;
//    medianBlur(frame, blurredFrame, 11);
//    TICK_ACCUMULATOR_END(blur);
    
//    TICK_ACCUMULATOR_START(threshold);
//    Mat thresholdFrame;
//    threshold(frame, thresholdFrame, 200, 255, CV_THRESH_BINARY);
//    TICK_ACCUMULATOR_END(threshold);

    TICK_ACCUMULATOR_START(detect);
    vector<KeyPoint> keyPoints;
    detector.detect(frame, keyPoints); 
    TICK_ACCUMULATOR_END(detect);

    TICK_ACCUMULATOR_START(sort);
    sort(keyPoints.begin(), keyPoints.end(),
         [] (KeyPoint const& a, KeyPoint const& b) { return a.size > b.size; });
    TICK_ACCUMULATOR_END(sort);

    TICK_ACCUMULATOR_START(filter);
    filterKeyPoints(keyPoints, hits, hitRects, skips);
    TICK_ACCUMULATOR_END(filter);
}

/**
 * Filter the keypoints looking for potential keypoints that
 * correspond to the two matching targets. Once found, the
 * target keypoints will be returns in the "hits" vector.i
 * Any keypoints skipped up until that point (because they
 * are not viable target candidates) will be returned in the
 * "skips" vector.
 */
void filterKeyPoints(vector<KeyPoint> const& keyPoints,
                     vector<vector<Point>>& hits,
                     vector<Rect>& hitRects,
                     vector<vector<Point>>& skips)
{
    if (keyPoints.size() == 0)
    {
        return;
    }

    if (keyPoints.size() == 1)
    {
        vector<Point> points;
        Rect rect;

        keyPointToPointsAndRect(*(keyPoints.begin()), points, rect);

        skips.push_back(points);
        return;
    }

    for (auto iter = keyPoints.begin(); iter != keyPoints.end()-1; iter++)
    {
        for (auto other = iter+1; other != keyPoints.end(); other++)
        {
            auto avgTargetSize = ((*iter).size + (*other).size)/2;
            auto deltaX = abs((*iter).pt.x - (*other).pt.x);
            auto deltaY =  abs((*iter).pt.y - (*other).pt.y);
            if ((deltaX < 9 * avgTargetSize) && (deltaX > 2 * avgTargetSize) &&
                (deltaY < 3 * avgTargetSize))
            {
                vector<Point> iterPoints;
                Rect iterRect;

                keyPointToPointsAndRect(*iter, iterPoints, iterRect);
                hits.push_back(iterPoints);
                hitRects.push_back(iterRect);

                vector<Point> otherPoints;
                Rect otherRect;

                keyPointToPointsAndRect(*other, otherPoints, otherRect);
                hits.push_back(otherPoints);
                hitRects.push_back(otherRect);

                return;
            }
            else
            {
                vector<Point> points;
                Rect rect;

                keyPointToPointsAndRect(*iter, points, rect);
                skips.push_back(points);
            }
        }   
    }
}

/**
 * Filter the contours...
 */
void filterContours(vector<vector<Point>> const& contours,
                    vector<vector<Point>>& hits,
                    vector<Rect>& hitRects,
                    vector<vector<Point>>& skips)
{
    if (contours.size() == 0)
    {
        return;
    }

    if (contours.size() == 1)
    {
        skips.push_back(*(contours.begin()));
        return;
    }

    for (auto iter = contours.begin(); iter != contours.end()-1; iter++)
    {
        Rect rect = boundingRect(*iter);
        if (rect.height < 1 * rect.width || rect.height > 4 * rect.width)
        {
//            cout << "Skipping contour (aspect ratio)..." << endl;
            continue;
        }

        if (rect.area() > 6000)
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

            if (rectOther.area() > 6000)
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
    params.thresholdStep = 10;              // 10
    params.minThreshold = 50;               // 50
    params.maxThreshold = 220;              // 220
    params.minRepeatability = 2;            // 2
    params.minDistBetweenBlobs = 10;        // 10
    params.filterByColor = true;            // true
    params.blobColor = 255;                 // 255
    params.filterByArea = true;             // true
    params.minArea = 1000 * fsfs;           // 1000
    params.maxArea = 20000 * fsfs;          // INT_MAX
    params.filterByCircularity = true;      // true
    params.minCircularity = 0;              // 0
    params.maxCircularity = 1;              // 1
    params.filterByInertia = true;          // true
    params.minInertiaRatio = 0.1;           // 0.1
    params.maxInertiaRatio = INT_MAX;       // INT_MAX
    params.filterByConvexity = true;        // true
    params.minConvexity = 0.95;             // 0.95
    params.maxConvexity = INT_MAX;          // INT_MAX

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
