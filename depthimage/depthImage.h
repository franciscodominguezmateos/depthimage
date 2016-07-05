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
	Mat dImg;
	Mat cImg;
	float fx,fy;
	float cx,cy;
	float level; //pyramid level
	float factor;//Sturm data set is 5000
	Point3f centroid;
	vector<string> getLinesFromFile(string fileName);
	vector<string> split(string line);
public:
	DepthImage();
	DepthImage(string basepath,int nImg);
	virtual ~DepthImage();
	inline Vec3b getColor(int u,int v){return cImg.at<Vec3b>(v,u);}
	inline Vec3b getColor(Point2f p){return getColor((int)p.x,(int)p.y);}
	inline float getDepth(int u,int v){return dImg.at<float>(v,u)/factor;}
	inline float getDepth(Point2f p){return getDepth((int)p.x,(int)p.y);}
	Point3f getPoint3D(int u,int v);
	Point3f getPoint3D(Point2f p){return getPoint3D((int)p.x,(int)p.y);}
	Point3f getPoint3Ddeep(int u,int v,float deep);
	Point2f project(Point3f p);
	bool is2DPointInImage(Point2f p);
	float projectiveDistance(Point3f p);
	inline bool isGoodDepthPixel(int u,int v){float d=dImg.at<float>(v,u);return d>1e-6;}//d==0 bad
	inline bool isGoodPoint3D(Point3f p){return p.z>0.001;}//Z==0 bad
	vector<Point3f> getPoints3D();
	vector<Vec3b> getColors();
	vector<Point3f> getPoints3DCentered();
	inline const Mat& getImg() const {	return cImg;}
	inline void setImg(const Mat& img) {cImg = img;}
	inline float getCx() const {return cx;	}
	inline void setCx(float cx) {this->cx = cx;}
	inline float getCy() const {return cy;	}
	inline void setCy(float cy) {this->cy = cy;}
	inline const Mat& getDepth() const {return dImg;	}
	void setDepth(const Mat& img);
	inline float getFactor() const {return factor;	}
	inline void setFactor(float factor) {this->factor = factor;}
	inline float getFx() const {return fx;	}
	inline void setFx(float fx) {this->fx = fx;}
	inline float getFy() const {return fy;	}
	inline void setFy(float fy) {this->fy = fy;}
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

	void glRender();
};

#endif /* DEPTHIMAGE_H_ */
