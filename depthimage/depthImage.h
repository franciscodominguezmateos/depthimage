/*
 * depthImage.h
 *
 *  Created on: Oct 17, 2015
 *      Author: Francisco Dominguez
 */

#ifndef DEPTHIMAGE_H_
#define DEPTHIMAGE_H_
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <GL/glut.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class DepthImage {
	string timestamp;
	Mat dImg;
	Mat cImg;
	Mat gImg;//gray image float
	Mat gXImg;
	Mat gYImg;
	// a CV_32FC3 normal image
	Mat nImg;
	Mat mImg;// Mask image. Do I need it?
	//Rt transformation for this camera
	Mat R;
	Mat t;
	float fx,fy;
	float cx,cy;
	Mat K;//Intrinsic Matrix
	float level; //pyramid level
	float factor;//Sturm data set is 5000
	Point3f centroid;
	vector<string> getLinesFromFile(string fileName);
	vector<string> split(string line);
public:
	void setCameraMatrixFreiburg1(){
	    fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
	    computeK();
	}
	void setCameraMatrixFreiburg2(){
	    fx = 520.9f; fy = 521.0f; cx = 325.1f; cy = 249.7f;
	    computeK();
	}
	DepthImage();
	DepthImage(string basepath,int nImg);
	virtual ~DepthImage();
	// u is column or X axis, v is row or Y axis
	inline Vec3b getColor(int u,int v){return cImg.at<Vec3b>(v,u);}
	inline Vec3b getColor(Point2f p){return getColor((int)p.x,(int)p.y);}
	inline Point3f getNormal(int u,int v){return nImg.at<Vec3f>(v,u);}
	inline Point3f getNormal(Point2f p){return getNormal((int)p.x,(int)p.y);}
	inline Point3f getNormalGlobal(int u,int v){
		Point3f p=getNormal(u,v);
	    Mat mp=(Mat_<double>(3, 1) << p.x, p.y, p.z);
	    Mat tp=R*mp;
	    return Point3f(tp.at<double>(0,0),tp.at<double>(1,0),tp.at<double>(2,0));
	}
	inline Point3f getNormalGlobal(Point2f p){return getNormalGlobal((int)p.x,(int)p.y);}
	inline void  setColor(int u,int v,Vec3b c){cImg.at<Vec3b>(v,u)=c;}
	inline float getDepth(int u,int v){return dImg.at<float>(v,u);}
	inline float getDepth(Point2f p){return getDepth((int)p.x,(int)p.y);}
	inline void  setDepth(int u,int v,float d){dImg.at<float>(v,u)=d;}
	inline float getGray(int u,int v){return gImg.at<float>(v,u);}
	inline float getGray(Point2f p){return getGray((int)p.x,(int)p.y);}
	Point3f getPoint3D(int u,int v);
	Point3f getGlobalPoint3D(int u,int v);
	Point3f getPoint3D(Point2f p){return getPoint3D((int)p.x,(int)p.y);}
	Point3f getPoint3Ddeep(int u,int v,float deep);
	Point2f project(const Point3f &p);
	Point2f projectGlobal(Point3f &p);
	bool is2DPointInImage(Point2f &p);
	float projectiveDistance      (Point3f &p);
	float projectiveDistanceGlobal(Point3f &pg);
	inline bool isGoodDepthPixel(Point2f &p){int u=p.x;int v=p.y;return isGoodDepthPixel(u,v);}//d==0 bad
	inline bool isGoodDepthPixel(int u,int v){float d=dImg.at<float>(v,u);return d>1e-6;}//d==0 bad
	inline bool isGoodPoint3D(Point3f p){return p.z>0.0001;}//Z==0 bad
	vector<Point3f> getPoints3D();
	vector<Point2f> getPoints2D();
	vector<Point3f> getGlobalPoints3D();
	vector<Vec3b> getColors();
	vector<Point3f> getPoints3DCentered();
	inline const Mat& getImg() const {	return cImg;}
	inline const Mat& getGray() const {return gImg;}
	inline const Mat& getGradXImg() const {return gXImg;}
	inline const Mat& getGradYImg() const {return gYImg;}
	inline void setImg(const Mat& img) {
		cImg = img;
		//cvtColor(img,gImg,CV_BGR2GRAY);
		Mat gtemp;
		cvtColor(img,gtemp,CV_BGR2GRAY);
		gtemp.convertTo(gImg,CV_32F);
		gImg/=255;
	}
	inline const Mat& getNormals() const {return nImg;}
	inline const Mat& getDepth() const {return dImg;	}
	void setDepth(const Mat& img);
	inline void computeK(){
		K = (Mat_<double>(3, 3) << fx   ,  0.00, cx,
				                   0.00,  fy   , cy,
								   0.00,  0.00, 1.00);
	}
	inline float getCx() const {return cx;	}
	inline void setCx(float cx) {this->cx = cx;computeK();}
	inline float getCy() const {return cy;	}
	inline void setCy(float cy) {this->cy = cy;computeK();}
	inline float getFx() const {return fx;	}
	inline void setFx(float fx) {this->fx = fx;computeK();}
	inline float getFy() const {return fy;	}
	inline void setFy(float fy) {this->fy = fy;computeK();}
	inline Mat getK(){return K;}
	inline float getFactor() const {return factor;	}
	inline void setFactor(float factor) {this->factor = factor;}
	inline float getLevel() const {return level;}
	inline void setLevel(float level) {this->level = level;}
	inline int cols(){return cImg.cols;}
	inline int rows(){return cImg.rows;}
	inline int size(){return cImg.cols*cImg.rows;}
	inline Point3f getCentroid(){return centroid;}
	DepthImage sparse();
    void bilateralDepthFilter(){
        Mat depth = Mat(dImg.size(), CV_32FC1, Scalar(0));
        const double depth_sigma = 50;// 0.03;
        const double space_sigma = 4.5;  // in pixels
        Mat invalidDepthMask = dImg == 0.f;
        dImg.setTo(-5*depth_sigma, invalidDepthMask);
        bilateralFilter(dImg, depth, -1, depth_sigma, space_sigma);
        depth.setTo(0.f, invalidDepthMask);
        setDepth(depth);
    }
    void getTransform(vector<string> l);
    inline Mat  getR(){return R;}
    inline Mat  getT(){return t;}
    inline void setR(Mat Rp){Rp.copyTo(R);}
    inline void setT(Mat tp){tp.copyTo(t);}
    inline void setPose(Mat Rp,Mat tp){setR(Rp);setT(tp);}
    Point3f toGlobal(Point3f &p){
    	Mat mp=(Mat_<double>(3, 1) << p.x, p.y, p.z);
    	//cout << "mp"<< mp << endl;
    	Mat tp=R*mp+t;
    	//cout << "tp"<< tp << endl;
    	return Point3f(tp.at<double>(0,0),tp.at<double>(1,0),tp.at<double>(2,0));
    }
    Point3f toLocal(Point3f &p){
    	Mat mp=(Mat_<double>(3, 1) << p.x, p.y, p.z);
    	//cout << "mp"<< mp << endl;
    	Mat tp=R.t()*(mp-t);
    	//cout << "tp"<< tp << endl;
    	return Point3f(tp.at<double>(0,0),tp.at<double>(1,0),tp.at<double>(2,0));
    }
    //new size of the image s=0.5 is halft and 1.0 the same just clone the
    DepthImage pyrDown(float s){
    	DepthImage di;
    	Size sz(cImg.cols*s,cImg.rows*s);
    	cv::pyrDown(cImg,di.cImg,sz);
    	cv::pyrDown(gImg,di.gImg,sz);
    	//cv::pyrDown(dImg,di.dImg,sz);
    	//Not so easy to pyrDown depth image dImg,
    	//it is compulsory to reset 0 depths
        //this below doesn't work
    	di.dImg=Mat::zeros(sz, dImg.type());

    	for(float i=0;i<dImg.rows;i+=1.0/s)
    		for(float j=0;j<dImg.cols;j+=1.0/s){
    		    float x=j*s;
    			float y=i*s;
    			//if(getDepth(j,i)<0.001)
    			di.dImg.at<float>(y,x)=getDepth(j,i);
    		}

    	if(!gXImg.empty())
    		cv::pyrDown(gXImg,di.gXImg,sz);
    	if(!gYImg.empty())
    		cv::pyrDown(gYImg,di.gYImg,sz);
    	if(!nImg.empty())
    		cv::pyrDown(nImg,di.nImg,sz);
    	di.fx*=s;
    	di.fy*=s;
    	di.cx*=s;
    	di.cy*=s;
    	R.copyTo(di.R);
    	t.copyTo(di.t);
    	return di;
    }
    void computeGrad(){
    	  int scale = 1;
    	  int delta = 0;
    	  int ddepth = CV_32F;

    	  /// Convert it to gray
    	  Mat gray;
    	  //Gaussian blur in order to avoid picks
    	  GaussianBlur( gImg, gray, Size(3,3), 0, 0, BORDER_DEFAULT );

    	  /// Gradient X
    	  Scharr( gray, gXImg, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    	  //Sobel( gray, gXImg, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

    	  /// Gradient Y
    	  Scharr( gray, gYImg, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    	  //Sobel( gray, gYImg, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    }
    void computeNormals(){
    	Mat depth=dImg;
    	if(depth.type() != CV_32FC1)
    	        depth.convertTo(depth, CV_32FC1);

    	Mat normals(depth.size(), CV_32FC3);
    	Vec3f n;

    	for(int x = 0; x < depth.rows; ++x)
    	{
    	    for(int y = 0; y < depth.cols; ++y)
    	    {
    	        // use float instead of double otherwise you will not get the correct result
    	        // check my updates in the original post. I have not figure out yet why this
    	        // is happening.
    	    	float dx1y=depth.at<float>(x+1, y);
    	    	float dx0y=depth.at<float>(x-1, y);
    	    	float dxy1=depth.at<float>(x, y+1);
    	    	float dxy0=depth.at<float>(x, y-1);
    	    	//Are all valid depth values
    	    	if(dx1y>0.01 && dx0y>0.01 && dxy1>0.01 &&dxy0>0.0){
					float dzdx = ( dx1y- dx0y) / 2.0;
					float dzdy = ( dxy1- dxy0) / 2.0;
    	    		// could be worth check if they are not big differences
					if(abs(dzdx)<0.1 && abs(dzdy)<0.1){
						Vec3f d(-dzdx, -dzdy, 1.0f);
						n = normalize(d);
					}
					else{
						cout << "dzdx="<<dzdx<<",dzdy="<<dzdy<<endl;
						n=Vec3f(0,0,0);
					}
    	    	}
    	    	else{
    	    		n=Vec3f(0,0,0);
    	    	}
    	        normals.at<Vec3f>(x, y) = n;
    	    }
    	}
    	nImg=normals;
    }
	void glRender();
};

#endif /* DEPTHIMAGE_H_ */
