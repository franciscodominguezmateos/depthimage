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
DepthImage::DepthImage(string basepath,int nImg):DepthImage(){
	string assopath,imagepath,depthpath;
	assopath=basepath+"/association.txt";
	vector<string> lines=getLinesFromFile(assopath);
	vector<string> words=split(lines[nImg]);
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
	this->dImg=depth;
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
Point3f DepthImage::getPoint3Ddeep(int u,int v,float deep){
	float x=u;
	float y=v;
	float Z=deep;
	float X = (x - cx) * Z / fx;
	float Y = (y - cy) * Z / fy;
	return Point3f(X,Y,Z);
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
				Point3f p=getPoint3D(u,v);
				glVertex3f(p.x,-p.y,-p.z);
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
