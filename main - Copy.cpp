#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <vector>
#include <list>
using namespace cv;
using namespace std;


int main()
{
	FILE *pf;
	pf = fopen("siftmatching.txt","w");
	int nframes = 255;

	vector<vector<int>> indexmat;
	vector<vector<int>> indexpts;
	int cnt = 0;
	//vector<int> tmp_indexpts;
	//vector<int> add_indexpts;
	//vector<int> tmp_indexmat;
	//vector<int> add_indexmat;
	
	Mat idx_mat = Mat::zeros(nframes,100000,CV_8UC1);
	//char fname1[100];
	char fname2[100];
	char fnour[100];
	char fname1[] = "C:\\X_WORK\\experiment\\movingCamera\\ECCV2012_multi_scale_clustering\\msam_msmc\\book\\1.jpg";
	Mat img1 = imread(fname1);
	int sz = 0;
	if(img1.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}

	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	vector<vector<KeyPoint>> KeyPoints(nframes);
	KeyPoints[0] = keypoints1;
	vector<vector<int>> points_position(nframes);

	//Mat tidx, aidx;
	
	//indexmat[0].reserve(keypoints1.size());
	vector<int> indexpts1(keypoints1.size(),-1);
	vector<int> indexmat1(keypoints1.size(),-1);

	for (int i = 2; i <=nframes-200; i ++)
	{
		sprintf(fname2,"C:\\X_WORK\\experiment\\movingCamera\\ECCV2012_multi_scale_clustering\\msam_msmc\\book\\%d.jpg",i);
		Mat img2 = imread(fname2);
		if(img2.empty())
		{
			printf("Can't read one of the images\n");
			return -1;
		}

		// detecting keypoints
		detector.detect(img2, keypoints2);

		// computing descriptors
		extractor.compute(img2, keypoints2, descriptors2);
		vector<int> indexpts2(keypoints2.size(),-1);
		vector<int> indexmat2(keypoints2.size(),-1);

		//indexmat[i-1].reserve(keypoints2.size());
		//indexpts1.reserve(keypoints2.size());
		//indexmat1.reserve(keypoints2.size());

		// matching descriptors
		BFMatcher matcher(NORM_L2);
		vector<DMatch> matches;
		matcher.match(descriptors1, descriptors2, matches);

		// drawing the results
		namedWindow("matches", 1);
		Mat img_matches;
		drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
		imshow("matches", img_matches);
		waitKey(100);
		KeyPoints.push_back(keypoints2);

		////////////////////////////////////////////////////////////////
		
		if (i<=2)
		{
			for (int j = 0; j <matches.size();j++)
			{
				indexpts1[j] = (matches[j].queryIdx);
				indexmat1[matches[j].queryIdx]=j;
				indexpts2[j] = (matches[j].trainIdx);
				indexmat2[matches[j].trainIdx]=j; //keypoints position in matrix
				cnt = cnt +1;
			}
			
			
		} 
		else
		{
			
			for (int j = 0; j <matches.size();j++)
			{
				indexpts1[j] = matches[j].queryIdx;
				indexpts2[j] = matches[j].trainIdx;

				if (indexmat1[matches[j].queryIdx]<0)
				{
					cnt = cnt+1;
					indexmat1[matches[j].queryIdx] = cnt;
					indexmat2[matches[j].trainIdx] = cnt;
				} 
				else
				{
					indexmat2[matches[j].trainIdx] = indexmat1[matches[j].queryIdx];
				}				

			}


		}
		indexmat.push_back(indexmat1);
		if (i == nframes)
		{
			indexmat.push_back(indexmat2);
		}
		
		//tmp_indexpts.clear();
		//tmp_indexpts = indexpts2;
		indexpts1.clear();
		indexpts1 = indexpts2;
		indexpts2.clear();
		indexmat1.clear();
		indexmat1 = indexmat2;
		indexmat2.clear();
		//////////////////////////////////////////////
		//sz = matches.size()+sz;


		keypoints1.clear();
		keypoints1 = keypoints2;
		keypoints2.clear();
		descriptors1 = descriptors2;		
		//delete index1;
		//delete index2;
	/////////////////////////////
		////%%%%%%%		fprintf(pf,"%.4f %.4f ",keypoints1[(index1[j])].pt.x,keypoints1[(index1[j])].pt.y);	fprintf(pf,"\n");
	}
	fclose(pf);
	

	return 0;
}
