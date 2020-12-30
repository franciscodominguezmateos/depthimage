/*
 * depthImage.h
 *
 *  Created on: Oct 17, 2015
 *      Author: Francisco Dominguez
 *  8/12(2020 refurbished
 */
#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <GL/glut.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// This library need to be linked with libfrennect too
#include "libfreenect.h"
#include "libfreenect_sync.h"

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
public:
	void setCameraMatrixFreiburg1(){
	    fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
	    computeK();
	}
	void setCameraMatrixFreiburg2(){
	    fx = 520.9f; fy = 521.0f; cx = 325.1f; cy = 249.7f;
	    computeK();
	}
	vector<string> getLinesFromFile(string fileName){
		vector<string> lines;
		ifstream myfile(fileName.c_str());
		string str;
		while (getline(myfile,str))
			lines.push_back(str);
		return lines;
	}
	vector<string> split(string line){
		string str;
		istringstream ss(line);
		vector<string> words;
		while(ss>> str)
			words.push_back(str);
		return words;
	}
	/*
	def transform44(l):
	"""
	Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

	Input:
	l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
	     (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

	Output:
	matrix -- 4x4 homogeneous transformation matrix
	"""
	t = l[1:4]
	q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
	nq = numpy.dot(q, q)
	if nq < _EPS:
	    return numpy.array((
	    (                1.0,                 0.0,                 0.0, t[0])
	    (                0.0,                 1.0,                 0.0, t[1])
	    (                0.0,                 0.0,                 1.0, t[2])
	    (                0.0,                 0.0,                 0.0, 1.0)
	    ), dtype=numpy.float64)
	q *= numpy.sqrt(2.0 / nq)
	q = numpy.outer(q, q)
	return numpy.array((
	    (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
	    (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
	    (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
	    (                0.0,                 0.0,                 0.0, 1.0)
	    ), dtype=numpy.float64)

	*/
	void quaternions2Rt(double tx,double ty,double tz,double qx,double qy,double qz,double qw){
		double nq=qx*qx + qy*qy + qz*qz + qw*qw;
		qx*=sqrt(2.0/nq);
		qy*=sqrt(2.0/nq);
		qz*=sqrt(2.0/nq);
		qw*=sqrt(2.0/nq);
		R  = (Mat_<double>(3, 3) << 1.0-qy*qy-qz*qz,    qx*qy-qz*qw,    qx*qz+qy*qw,
				                        qx*qy+qz*qw,1.0-qx*qx-qz*qz,    qy*qz-qx*qw,
										qx*qz-qy*qw,    qy*qz+qx*qw,1.0-qx*qx-qy*qy);
		t  = (Mat_<double>(3, 1) << tx, ty, tz);
	}
	void getTransform(vector<string> l){
		int pidx=4;
		double tx=atof(l[pidx+1].c_str());
		double ty=atof(l[pidx+2].c_str());
		double tz=atof(l[pidx+3].c_str());
		double qx=atof(l[pidx+4].c_str());
		double qy=atof(l[pidx+5].c_str());
		double qz=atof(l[pidx+6].c_str());
		double qw=atof(l[pidx+7].c_str());
		quaternions2Rt(tx,ty,tz,qx,qy,qz,qw);
	}
	DepthImage() {
		level=1;
		//Intrinsic data for Stum dataset
		fx = 525.0/level;  // focal length x
		fy = 525.0/level;  // focal length y
		cx = 319.5/level;  // optical center x
		cy = 239.5/level;  // optical center y
		setCameraMatrixFreiburg1();
		factor = 5000.0; // for the 16-bit PNG files
	}
	//Freenect_sync DepthImage capture
	// int i is the device number
	// Images are not alligned
	void no_kinect_quit(void)
	{
	    printf("Error: Kinect not connected?\n");
	    exit(1);
	}
	DepthImage(int i){
		R  = (Mat_<double>(3, 3) << 1., 0., 0.,
				                    0., 1., 0.,
									0., 0., 1.);
		t  = (Mat_<double>(3, 1) << 0., 0., 0.);
	    short *depth = 0;
	    char *rgb = 0;
	    uint32_t ts;
	    if (freenect_sync_get_depth((void**)&depth, &ts, 0, FREENECT_DEPTH_11BIT) < 0)
		no_kinect_quit();
	    if (freenect_sync_get_video((void**)&rgb, &ts, 0, FREENECT_VIDEO_RGB) < 0)
		no_kinect_quit();

	    Mat cv_image = Mat( 480, 640, CV_8UC3);
	    Mat cv_depth = Mat( 480, 640, CV_16UC1);
	    // 640x480x 2= word CV_16UC1
	    memcpy(depth, cv_depth.data, 640*480*2);
	    // 3= CV_8UC3
	    memcpy(rgb, cv_depth.data, 640*480*3);
	    cv_depth.convertTo(cv_depth, CV_32F);
		this->setImg(cv_image);
		//this->dImg=depth;
		//cv_depth/=factor;
		this->setDepth(cv_depth);
	}
	DepthImage(string basepath,int nImg):DepthImage(){
		string assopath,imagepath,depthpath;
		R  = (Mat_<double>(3, 3) << 1., 0., 0.,
				                    0., 1., 0.,
									0., 0., 1.);
		t  = (Mat_<double>(3, 1) << 0., 0., 0.);
		assopath=basepath+"/associationgt.txt";
		vector<string> lines=getLinesFromFile(assopath);
		vector<string> words=split(lines[nImg]);
		if(words.size()==12)
			getTransform(words);
		timestamp=words[0];
	    imagepath=basepath +"/"+words[1];
	    depthpath=basepath +"/"+words[3];
	    Mat image = imread(imagepath, IMREAD_COLOR );
	    Mat depth = imread(depthpath, IMREAD_ANYDEPTH );
	    if ( !image.data ) {
	        printf("No image data \n");
	    }
	    if ( !depth.data ) {
	        printf("No depth data \n");
	    }
	    depth.convertTo(depth, CV_32F);
		this->setImg(image);
		//this->dImg=depth;
		depth/=factor;
		this->setDepth(depth);
	}
	~DepthImage() {
		// TODO Auto-generated destructor stub
	}
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
	// u is column or X axis, v is row or Y axis
	inline Vec3b   getColor(int u,int v){return cImg.at<Vec3b>(v,u);}
	inline Vec3b   getColor(Point2f p){return getColor((int)p.x,(int)p.y);}
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
	inline Point3f getGlobalPoint3D(int u,int v){
		Point3f p=getPoint3D(u,v);
		return toGlobal(p);
	}
	inline Point3f getPoint3Ddeep(int u,int v,float deep){
		float x=u;
		float y=v;
		float &Z=deep;
		float X = (x - cx) * Z / fx;
		float Y = (y - cy) * Z / fy;
		return Point3f(X,Y,Z);
	}
	Point3f getPoint3D(int u,int v){
		return getPoint3Ddeep(u,v,dImg.at<float>(v,u));
	}
	Point3f getPoint3D(Point2f p){return getPoint3D((int)p.x,(int)p.y);}
	Point2f project(const Point3f &p){
		const float &X=p.x;
		const float &Y=p.y;
		const float &Z=p.z;
		float x=X*fx/Z+cx;
		float y=Y*fy/Z+cy;
		return Point2f(x,y);
	}
	Point2f projectGlobal(Point3f &pg){
		Point3f p=toLocal(pg);
		return project(p);
	}
	// pg is a global 3D point
	float projectiveDistanceGlobal(Point3f &pg){
		Point3f p=toLocal(pg);
		return projectiveDistance(p);
	}
	// p is a local 3D point
	float projectiveDistance(Point3f &p){
		Point2f p2D=project(p);
		if (this->is2DPointInImage(p2D)){//Does it project in image?
			int u=p2D.x;
			int v=p2D.y;
			if(isGoodDepthPixel(u,v)){//is there depth information of it?
				float d=this->getDepth(u,v);
				// negative value is in front of the surface
				// positive value is behind the surface
				float pd=p.z-d;
				if(-0.5<pd && pd<0.5){//Is it too far or too near?
					//cout << "pd="<<pd << "\td=" << d <<" x=" << p.x << " y=" << p.y << " z=" << p.z <<" u=" << u << " v=" << v <<endl;
				    return pd;
				}
			}
		}
		return 1e32; //infinity a big number
	}
	inline bool isGoodDepthPixel(Point2f &p){int u=p.x;int v=p.y;return isGoodDepthPixel(u,v);}//d==0 bad
	inline bool isGoodDepthPixel(int u,int v){float d=dImg.at<float>(v,u);return d>1e-6;}//d==0 bad
	inline bool isGoodPoint3D(Point3f p){return p.z>0.0001;}//Z==0 bad
	vector<Point2f> getPoints2D(){
		vector<Point2f> vp;
		for (int v=0;v<dImg.rows;v++){
			for (int u=0;u<dImg.cols;u++){
				if (isGoodDepthPixel(u,v)){
					Point2f p=Point2f(u,v);
					vp.push_back(p);
				}
			}
		}
		return vp;
	}
	vector<Point3f> getGlobalPoints3D(){
		vector<Point3f> vp;
		for (int v=0;v<dImg.rows;v++){
			for (int u=0;u<dImg.cols;u++){
				if (isGoodDepthPixel(u,v)){
					Point3f p=getGlobalPoint3D(u,v);
					vp.push_back(p);
				}
			}
		}
		return vp;
	}
	vector<Vec3b> getColors(){
		vector<Vec3b> vp;
		for (int v=0;v<dImg.rows;v++){
			for (int u=0;u<dImg.cols;u++){
				if (isGoodDepthPixel(u,v)){
					Vec3b p=getColor(u,v);
					vp.push_back(p);
				}
			}
		}
		return vp;
	}
	vector<Point3f> getPoints3DCentered(){
		vector<Point3f> vp;
		Point3f centroid=getCentroid();
		for (int v=0;v<dImg.rows;v++){
			for (int u=0;u<dImg.cols;u++){
				if (isGoodDepthPixel(u,v)){
					Point3f p=getPoint3D(u,v);
					p-=centroid;
					vp.push_back(p);
				}
			}
		}
		return vp;
	}
	void setDepth(const Mat& img) {
		dImg = img;
		vector<Point3f> pts=this->getPoints3D();
		Point3f pt;
		for(Point3f p:pts){
			pt+=p;
		}
		centroid=pt;
		centroid.x/=pts.size();
		centroid.y/=pts.size();
		centroid.z/=pts.size();
		computeNormals();
	}
	void glRender(){
		glPointSize(2.0);
		glBegin(GL_POINTS);
		for (int v=0;v<dImg.rows;v++)
		{
			for (int u=0;u<dImg.cols;u++)
			{
				if(isGoodDepthPixel(u,v)){
				    Vec3b col=getColor(u,v);
					float b=col.val[0]/255.0;
					float g=col.val[1]/255.0;
					float r=col.val[2]/255.0;
					glColor3f(r,g,b);
					Point3f p=getGlobalPoint3D(u,v);
					//cout << "x=" << p.x << " y=" << p.y << " z=" << p.z  <<endl;
					//glVertex3f(p.x,-p.y,-p.z);
					glVertex3f(p.x,p.y,p.z);
				}
			}
		}
		glEnd();
	}
	DepthImage sparse(){
		//Canny params
		//int edgeThresh = 1;
		int lowThreshold=50;
		//int const max_lowThreshold = 100;
		int ratio = 3;
		int kernel_size = 5;

		DepthImage dImg=*this;
		DepthImage di;
		Mat src_gray,detected_edges,imagec,depth;
		imagec=dImg.getImg();
		/// Reduce noise with a kernel 3x3
		cvtColor( imagec, src_gray, CV_BGR2GRAY );
		//blur( src_gray, detected_edges, Size(3,3) );
		GaussianBlur( src_gray, detected_edges, Size(5,5), 5, 5, BORDER_DEFAULT );
		//medianBlur (src_gray, detected_edges, 15 );
		//bilateralFilter (src_gray, detected_edges, 15,15*2, 15/2 );
		/// Canny detector
		Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

		Mat depthAll;
		depthAll=dImg.getDepth();
	    depth = Scalar::all(0);
	    depthAll.copyTo( depth, detected_edges);

	    di.setImg(imagec);
	    di.setDepth(depth);
	//    cout << dImg.getCentroid()<< " centroid"<<endl;
	//    cout << dImg.getPoints3D().size()/1000 << "mil filtered points" <<endl;
	//    cout << di.getCentroid()<< " centroid"<<endl;
	//    cout << di.getPoints3D().size()/1000 << "mil filtered points" <<endl;
	    return di;
	}
	bool is2DPointInImage(Point2f &p){
		if (p.x<0)
			return false;
		if (p.y<0)
			return false;
		if (p.x>=dImg.cols)
			return false;
		if (p.y>=dImg.rows)
			return false;
		return true;
	}
	vector<Point3f> getPoints3D(){
		vector<Point3f> vp;
		for (int v=0;v<dImg.rows;v++){
			for (int u=0;u<dImg.cols;u++){
				if (isGoodDepthPixel(u,v)){
					Point3f p=getPoint3D(u,v);
					vp.push_back(p);
				}
			}
		}
		return vp;
	}
	inline const Mat& getImg()      const {return cImg;}
	inline const Mat& getGray()     const {return gImg;}
	inline const Mat& getGradXImg() const {return gXImg;}
	inline const Mat& getGradYImg() const {return gYImg;}
	inline const Mat& gerNormals()  const {return nImg;}
	Mat getImgDepth(){
		DepthImage &di=*this;
		Mat cimg=di.getImg();
		cimg.convertTo(cimg,CV_32FC3);
		cimg/=255;
		vector<Mat> channels;
		cv::split(cimg, channels);
		Mat dimg=di.getDepth();
		Mat img;
		img.convertTo(img,CV_32FC4);
		vector<Mat> vd = { channels[0], channels[1],channels[2], dimg };
		merge(vd, img);
		return img;
	}
	Mat getNormDepth(){
		DepthImage &di=*this;
		Mat nimg=di.getNormals();
		vector<Mat> channels;
		cv::split(nimg, channels);
		Mat dimg=di.getDepth();
		Mat img;
		img.convertTo(img,CV_32FC4);
		vector<Mat> vd = { channels[0], channels[1],channels[2], dimg };
		merge(vd, img);
		return img;
	}
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
	inline void computeK(){
		K = (Mat_<double>(3, 3) << fx   ,  0.00, cx,
				                   0.00,  fy   , cy,
								   0.00,  0.00, 1.00);
	}
	inline float getCx() const {return cx;	}
	inline void  setCx(float cx) {this->cx = cx;computeK();}
	inline float getCy() const {return cy;	}
	inline void  setCy(float cy) {this->cy = cy;computeK();}
	inline float getFx() const {return fx;	}
	inline void  setFx(float fx) {this->fx = fx;computeK();}
	inline float getFy() const {return fy;	}
	inline void  setFy(float fy) {this->fy = fy;computeK();}
	inline Mat getK(){return K;}
	inline float getFactor() const {return factor;	}
	inline void  setFactor(float factor) {this->factor = factor;}
	inline float getLevel() const {return level;}
	inline void  setLevel(float level) {this->level = level;}
	inline int cols(){return cImg.cols;}
	inline int rows(){return cImg.rows;}
	inline int size(){return cImg.cols*cImg.rows;}
	inline Point3f getCentroid(){return centroid;}
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
    inline Mat  getR(){return R;}
    inline Mat  getT(){return t;}
    inline void setR(Mat Rp){Rp.copyTo(R);}
    inline void setT(Mat tp){tp.copyTo(t);}
    inline void setPose(Mat Rp,Mat tp){setR(Rp);setT(tp);}
    //new size of the image s=0.5 is halft and 1.0 the same just clone the
    DepthImage pyrDown(float s){
    	DepthImage di;
    	Size sz(cImg.cols*s,cImg.rows*s);
    	cv::pyrDown(cImg,di.cImg,sz);
    	cv::pyrDown(gImg,di.gImg,sz);
    	//Not so easy to pyrDown depth image dImg,
    	//it is compulsory to reset 0 depths
        //this below doesn't work
    	di.dImg=Mat::zeros(sz, dImg.type());
    	//cv::pyrDown(dImg,di.dImg,sz);
    	//copy without filtering
    	for(float i=0;i<dImg.rows;i+=1.0/s)
    		for(float j=0;j<dImg.cols;j+=1.0/s){
				float x=j*s;
				float y=i*s;
				di.dImg.at<float>(y,x)=dImg.at<float>(i,j);
    		}
        //now set to zero all zero depth
    	for(float i=0;i<dImg.rows;i+=1.0/s)
    		for(float j=0;j<dImg.cols;j+=1.0/s){
    			if(getDepth(j,i)<0.001){
        		    float x=j*s;
        			float y=i*s;
        			di.dImg.at<float>(y,x)=0;
    			}
    		}
    	if(!gXImg.empty())
    		cv::pyrDown(gXImg,di.gXImg,sz);
    	if(!gYImg.empty())
    		cv::pyrDown(gYImg,di.gYImg,sz);
    	if(!nImg.empty())
    		cv::pyrDown(nImg,di.nImg,sz);
    	di.fx=fx*s;
    	di.fy=fy*s;
    	di.cx=cx*s;
    	di.cy=cy*s;
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
    inline bool isBadNormal(Point3f vn){
    	return isnan(vn.x) || isnan(vn.y) || isnan(vn.z);
    }
    inline bool isGoodNormal(int u,int v){
    	return !isBadNormal(nImg.at<Vec3f>(v, u));
    }
    void computeNormals(){
    	Mat depth=dImg;
    	if(depth.type() != CV_32FC1)
    	        depth.convertTo(depth, CV_32FC1);
    	Mat normals(depth.size(), CV_32FC3);
    	Vec3f n;
    	for(int x = 0; x < depth.cols; ++x){
    	    for(int y = 0; y < depth.rows; ++y){
    	    	if(isGoodDepthPixel(x,y)){
					// use float instead of double otherwise you will not get the correct result
					// check my updates in the original post. I have not figure out yet why this
					// is happening.
					float dx1y=depth.at<float>(y  ,x+1);
					float dx0y=depth.at<float>(y  ,x-1);
					float dxy1=depth.at<float>(y+1,x);
					float dxy0=depth.at<float>(y-1,x);
					float dxy =depth.at<float>(y  ,x);
					//Are all valid depth values
					if(dx1y>0.01 && dx0y>0.01 && dxy1>0.01 &&dxy0>0.01 && dxy>0.01){
						float dzdx = ( dx1y- dx0y) / 2.0;
						float dzdy = ( dxy1- dxy0) / 2.0;
						// could be worth check if they are not big differences
						if(abs(dzdx)<0.02 && abs(dzdy)<0.02){
							//Vec3f d(-dzdx, -dzdy, 0.005f);
				            // 3d pixels, think (x,y, depth)
				             /* * * * *
				              * * t * *
				              * * c l *
				              * * * * */
							//Vec3f t(y-1,x  ,dxy0);
							Vec3f t(getPoint3D(x  ,y-1));
							//Vec3f l(y  ,x-1,dx0y);
							Vec3f l(getPoint3D(x+1,y));
							//Vec3f c(y  ,x  ,dxy);
							Vec3f c(getPoint3D(x  ,y));
							Vec3f dx=l-c;
							Vec3f dy=t-c;
							Vec3f d = dx.cross(dy);
							n=normalize(d);
						}
						else{
							//cout << "dzdx="<<dzdx<<",dzdy="<<dzdy<<endl;
							n=Vec3f(0,0,0);
						}
					}
					else{
						n=Vec3f(0,0,0);
					}
    	    	}
    	    	else{
    	    		n=Vec3f(0,0,0);
    	    	}
    	        normals.at<Vec3f>(y, x) = n;
    	    }
    	}
    	for(int x = 0; x < depth.cols; ++x){
    	    for(int y = 0; y < depth.rows; ++y){
    	    	if(normals.at<Vec3f>(y,x)==Vec3f(0,0,0)){
    	    		depth.at<float>(y,x)=0;
    	    	}
    	    }
    	}
    	nImg=normals;
    }
};

