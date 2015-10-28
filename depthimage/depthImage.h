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
	Point3f getPoint3D(int u,int v);
	Point3f getPoint3Ddeep(int u,int v,float deep);
	inline bool isGoodDepthPixel(int u,int v){float d=dImg.at<float>(v,u);return d>1e-6;}//d==0 bad
	inline bool isGoodPoint3D(Point3f p){return p.z>0.001;}//Z==0 bad
	vector<Point3f> getPoints3D();
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
	void glRender();
};

#endif /* DEPTHIMAGE_H_ */
