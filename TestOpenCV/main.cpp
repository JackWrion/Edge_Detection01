#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void myROI(Mat img) {
	int lineType = LINE_8;

	//define 4 points of polygon mask
	Point mask[1][4];
	mask[0][0] = Point(50, 540);
	mask[0][1] = Point(960, 540);
	mask[0][2] = Point(800, 200);
	mask[0][3] = Point(300, 200);

	//fill polygon with white color
	const Point* ppt[1] = { mask[0] };
	int npt[] = { 4 };
	fillPoly(img,
		ppt,
		npt,
		1,
		Scalar(255, 255, 255),
		lineType);
}

void HoughlineDraw(Mat edges, Mat src) {
	
	vector<Vec4i> linesP;											// will hold the results of the detection
	HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 50, 10);			// runs the actual detection
	
	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	imshow("Detected Lines (in red)", src);
}

int main() {
	//import image from resource to 'img'
	string path = "Resources/road1.jpg";
	Mat img = imread(path);

	//resize img to qHD (960x540)
	Mat ImgResized;
	resize(img, ImgResized, Size(960,540));

	//convert ResizedImg to 1-channel color (Gray)
	Mat GrayImg;
	cvtColor(ImgResized, GrayImg, COLOR_BGR2GRAY);

	//Blur Image using Gaussian algorithm
	Mat BlurImg;
	GaussianBlur(GrayImg, BlurImg,Size(5,5),10);

	//Raw Edge-detection with Canny algorithm
	Mat edgesImg;
	Canny(BlurImg,edgesImg,50,200);

	//Create mask to define ROI
	Mat mask = Mat::zeros(540, 960, CV_8UC1);
	myROI(mask);
	
	//apply Mask on Edge-detected Image
	Mat FreshImg;
	bitwise_and(edgesImg, mask, FreshImg);

	//draw the road line to Image using Probabilistic Hough Line Transform
	HoughlineDraw(FreshImg, ImgResized);

	//display
	imshow("Original_Gray_Img", GrayImg);
	imshow("After_Masked_Img", FreshImg);
	imshow("Mask", mask);
	imshow("Edge_Detected_Img", edgesImg);
	
	waitKey(0);
	return 0;

}