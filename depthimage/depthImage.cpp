/*
 * depthImage.cpp
 *
 *  Created on: Oct 17, 2015
 *      Author: francisco
 */

#include "depthImage.h"

DepthImage::DepthImage() {
	level=1;
	//Intrinsic data for Stum dataset
	fx = 525.0/level;  // focal length x
	fy = 525.0/level;  // focal length y
	cx = 319.5/level;  // optical center x
	cy = 239.5/level;  // optical center y
	factor = 5000.0; // for the 16-bit PNG files
}
vector<string> DepthImage::getLinesFromFile(string fileName){
	vector<string> lines;
	ifstream myfile(fileName.c_str());
	string str;
	while (getline(myfile,str))
		lines.push_back(str);
	return lines;
}
vector<string> DepthImage::split(string line){
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
void DepthImage::getTransform(vector<string> l){
	int pidx=4;
	double tx=atof(l[pidx+1].c_str());
	double ty=atof(l[pidx+2].c_str());
	double tz=atof(l[pidx+3].c_str());
	double qx=atof(l[pidx+4].c_str());
	double qy=atof(l[pidx+5].c_str());
	double qz=atof(l[pidx+6].c_str());
	double qw=atof(l[pidx+7].c_str());
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
DepthImage::DepthImage(string basepath,int nImg):DepthImage(){
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
	this->cImg=image;
	//this->dImg=depth;
	this->setDepth(depth);
}
DepthImage::~DepthImage() {
	// TODO Auto-generated destructor stub
}
Point3f DepthImage::getPoint3D(int u,int v){
	float x=u;
	float y=v;
	float deep=dImg.at<float>(v,u);
	float Z=deep/factor;
	float X = (x - cx) * Z / fx;
	float Y = (y - cy) * Z / fy;
	return Point3f(X,Y,Z);
}
Point3f DepthImage::getGlobalPoint3D(int u,int v){
	Point3f p=getPoint3D(u,v);
	return toGlobal(p);
}

Point3f DepthImage::getPoint3Ddeep(int u,int v,float deep){
	float x=u;
	float y=v;
	float Z=deep;
	float X = (x - cx) * Z / fx;
	float Y = (y - cy) * Z / fy;
	return Point3f(X,Y,Z);
}
Point2f DepthImage::project(Point3f p){
	float X=p.x;
	float Y=p.y;
	float Z=p.z;
	float x=X*fx/Z+cx;
	float y=Y*fy/Z+cy;
	return Point2f(x,y);
}
float DepthImage::projectiveDistance(Point3f p){
	Point2f p2D=project(p);
	if (this->is2DPointInImage(p2D)){
		int u=p2D.x;
		int v=p2D.y;
		if(isGoodDepthPixel(u,v)){
			float d=this->getDepth(u,v);
			float pd=d-p.z;
			if(-0.1<pd && pd<0.1){
				//cout << "pd="<<pd << "\td=" << d <<" x=" << p.x << " y=" << p.y << " z=" << p.z <<" u=" << u << " v=" << v <<endl;
			    return pd;
			}
		}
	}
	return 1e32; //infinity a big number
}
bool DepthImage::is2DPointInImage(Point2f p){
	int u=p.x;
	int v=p.y;
	if (u<0)
		return false;
	if (v<0)
		return false;
	if (u>dImg.cols)
		return false;
	if (v>dImg.rows)
		return false;
	return true;
}
vector<Point3f> DepthImage::getPoints3D(){
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
vector<Point3f> DepthImage::getGlobalPoints3D(){
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
vector<Vec3b> DepthImage::getColors(){
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
vector<Point3f> DepthImage::getPoints3DCentered(){
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
void DepthImage::setDepth(const Mat& img) {
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
}
void DepthImage::glRender(){
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
DepthImage DepthImage::sparse(){
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
