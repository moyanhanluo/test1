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
	pf = fopen("siftmatching1.txt","w");
	int nframes = 25 ;//225;

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
	vector<int> indexpts1;
	vector<int> indexpts2;
	vector<int> indexmat1(keypoints1.size(),-1);
	
	for (int i = 2; i <=nframes; i ++)
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
		
		vector<int> indexmat2(keypoints2.size(),-1);

		//indexmat[i-1].reserve(keypoints2.size());
		//indexpts1.reserve(keypoints2.size());
		//indexmat1.reserve(keypoints2.size());

		// matching descriptors
		FlannBasedMatcher matcher;
		vector<DMatch> ini_matches,matches;
		matcher.match(descriptors1, descriptors2, ini_matches);


		double max_dist = 0;
		double min_dist = 400;


		for (int j = 0;j<descriptors1.rows;j++)
		{
			double dist = ini_matches[j].distance;
			if (dist<min_dist)
			{
				min_dist = dist;

			}
			if (dist>max_dist)
			{
				max_dist = dist;
			}
			

		}
		
		for (int j = 0;j<descriptors1.rows;j++)
		{
			if (ini_matches[j].distance<=max(2.5*min_dist,0.02))
			{
				matches.push_back(ini_matches[j]);
			}
			
		}
		

		// drawing the results
		//namedWindow("matches", 1);
		//Mat img_matches;
		//drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
		//imshow("matches", img_matches);
		//waitKey(100);
		KeyPoints[i-1] = keypoints2;

		////////////////////////////////////////////////////////////////
		
		if (i<=2)
		{
			for (int j = 0; j <matches.size();j++)
			{
				indexpts1.push_back((matches[j].queryIdx))  ;
				indexmat1[matches[j].queryIdx]=j;
				indexpts2.push_back((matches[j].trainIdx)) ;
				indexmat2[matches[j].trainIdx]=j; //keypoints position in matrix
				cnt = cnt +1;
			}
			
			
		} 
		else
		{
			
			for (int j = 0; j <matches.size();j++)
			{
				

				if (indexmat1[matches[j].queryIdx]<0)
				{
					cnt = cnt+1;
					indexmat1[matches[j].queryIdx] = cnt;
					indexmat2[matches[j].trainIdx] = cnt;
					//indexpts1.push_back(matches[j].queryIdx);
					//indexpts2.push_back(matches[j].trainIdx);
				} 
				else
				{
					indexmat2[matches[j].trainIdx] = indexmat1[matches[j].queryIdx];
					indexpts2.push_back((matches[j].trainIdx)) ;
					//indexpts2[j] = matches[j].trainIdx;

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
		cout<<"frame = "<< i <<"\n"<<endl;
		cout<<"------------------------"<<endl;
		//delete index1;
		//delete index2;
	/////////////////////////////
		////%%%%%%%		fprintf(pf,"%.4f %.4f ",keypoints1[(index1[j])].pt.x,keypoints1[(index1[j])].pt.y);	fprintf(pf,"\n");
	}
	vector<KeyPoint> kpts;
	vector<int> kidx;
	Point2f pos;
	for (int k = 0; k <nframes; k++)
	{
		kpts = KeyPoints[k];
		kidx = indexmat[k];
		
		sz = kpts.size();
		for (int l = 0; l <cnt; l++)
		{
			vector<int>::iterator itr = find(kidx.begin(),kidx.end(),l);
			if (itr == kidx.end())
			{
				pos.x = 0;
				pos.y = 0;
			} 
			else
			{
				int idx = itr - kidx.begin();
				pos.x = kpts[idx].pt.x;
				pos.y = kpts[idx].pt.y;
				if (pos.x<0||pos.y<0)
				{
					pos.x = 0;
					pos.y = 0;
				}
			}
			
			
			
			
			fprintf(pf,"%.2f %.2f ", pos.x,pos.y);
		}
		fprintf(pf,"\n");
		
		cout<<"frame = "<< k <<"\n"<<endl;
		cout<<"------------------------"<<endl;


	}
	
	fclose(pf);
	

	return 0;
}
