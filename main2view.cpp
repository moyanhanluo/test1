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
	
	int nframes = 225 ;//225;
	char fname2[100];
	char fnour[100];
	char fname1[] = "C:\\X_WORK\\experiment\\movingCamera\\ECCV2012_multi_scale_clustering\\msam_msmc\\book\\1.jpg";
	Mat img1 = imread(fname1);
	int h = -1;
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


	for (int i = 2; i <=nframes; i ++)
	{
		sprintf(fnour,"C:\\X_WORK\\experiment\\movingCamera\\ECCV2012_multi_scale_clustering\\msam_msmc\\book\\match_%d.txt",i-1);
		sprintf(fname2,"C:\\X_WORK\\experiment\\movingCamera\\ECCV2012_multi_scale_clustering\\msam_msmc\\book\\%d.jpg",i);
		FILE *pf;
		pf = fopen(fnour,"w");

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

		//vector<int> indexmat2(keypoints2.size(),-1);

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
			if (ini_matches[j].distance<=max(5*min_dist,0.02))
			{
				matches.push_back(ini_matches[j]);
				//cout<<h<<h<<endl;
				fprintf(pf,"%d %.2f %.2f %d %.2f %.2f \n", h,keypoints1[ini_matches[j].queryIdx].pt.x,keypoints1[ini_matches[j].queryIdx].pt.y,h,keypoints2[ini_matches[j].trainIdx].pt.x,keypoints2[ini_matches[j].trainIdx].pt.y);
			}

		}


		 //drawing the results
		namedWindow("matches", 1);
		Mat img_matches;
		drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
		imshow("matches", img_matches);
		waitKey(100);

		cout<<"frame="<<i<<"p_num"<<matches.size()<<endl;

		img1 = img2;
		keypoints1.clear();
		keypoints1 = keypoints2;
		keypoints2.clear();
		matches.clear();
		ini_matches.clear();
		descriptors1 = descriptors2;
		
	
		fclose(pf);
	}



	return 0;
}
