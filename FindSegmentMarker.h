#


#include "opencv2/opencv.hpp" 
#include <math.h>
#include "PanoStitcher.h"
#include "math.h"
#include <iostream>

typedef struct
{
	char key;		//线段编码：“0”“1”“-1”
	cv::Point ep0;	//线段端点0坐标(像素坐标)
	cv::Point ep1;	//线段端点1坐标（像素坐标）
	//cv::Point pi0;	//线段端点0坐标（像素坐标）
	//cv::Point pi1;	//线段端点1坐标（像素坐标）
}_KEY_SEGMENT;

#define minv(x,y) ( (x>y)?y:x)

class FindSegmentMarker
{
	public:
	//--------------------------------------------------------------构造函数--------------------------------------------------------//

	FindSegmentMarker(cv::Mat srcimage) //对应输入图像，输出，以及相机编码012, 向左（0）或者向右（1）
	{
		loadImg(srcimage);//载入图像
		//setMask();//设置mask(因为不太准确，先不用)

	}
	//--------------------------------------------------------------输出接口---------------------------------------------------------//
	void getResult(std::vector<_KEY_SEGMENT>&segments, bool f_or_b)//给出结果的接口
	{
		imgAdjust();//增强图像对比度

		grabCut(f_or_b);//粗提取
		imgThreshold();//二值化
		findContours();//找到轮廓
		setLSThreshold();//设置长短阈值


		for (int i = 0; i < exactContours.size(); i++)
		{
			_KEY_SEGMENT keyTem;
			keyTem.ep0 = detExtrePoi(exactContours[i]).first;
			keyTem.ep1 = detExtrePoi(exactContours[i]).second;
			//keyTem.key = judgeDis(keyTem.ep0, keyTem.ep1, shortThreshold, longThreshold);
			keyTem.key = judgeDis(keyTem.ep0, keyTem.ep1, i);
			segments.push_back(keyTem);
		}

		saveResult(segments);
	}
	//===============================================================各种提取数据的函数=======================================================//
	cv::Mat getimgOriginal()
	{
		return imgOriginal;
	}

	//cv::Mat getimgGreen()
	//{
	//	cv::split(imgOriginal,);
	//}

	cv::Mat getimgGray()
	{
		return imgForGray;
	}
	cv::Mat getimgForeground()
	{
		return imgForeground;
	}
	cv::Mat getimgBinary()
	{
		return imgBinary;
	}
	cv::Mat getimgSort()
	{
		imgOriginal.copyTo(imgSort);
		for (int i = 0; i < exactContours.size(); i++)
		{
			cv::drawContours(imgSort, exactContours, i, CV_RGB(i * 255 / exactContours.size(), 0, 0), CV_FILLED);
		}
		return imgSort;
	}


	private:

		//===================================================主要用来存储各种中间数据便于调试=================================================//
		bool f_or_b;//确定向左or向右找

		cv::Mat imgOriginal;//原始图像
		cv::Mat imgForeground;//粗提取后的图像
		cv::Mat imgForGray;//粗提取后转为灰度图像
		cv::Mat imgBinary;//粗体取后的图像进行二值化
		cv::Mat imgSort;//轮廓排序后图片，可从红色深度是否渐变判断
		cv::Mat imgSharpen;//锐化图片，增加对比度

		cv::Mat lineDiss;//用来存储每条直线长度的一维数组(validCon.size(), 1, CV_32FC1)
		cv::Mat mask;//存储粗提取需要的mask

		std::vector<std::vector<cv::Point>> roughContours;//粗提取轮廓
		std::vector<std::vector<cv::Point>> exactContours;//剔除无效轮廓后的有效轮廓
		
		double shortThreshold, longThreshold;//判断直线长度的两个阈值
		std::vector<double> shortTV, longTV;
		double lightThreshold;//图像二值化的阈值

		cv::Rect rectangle;//粗提取时的区域，坑1尽量用mask代替
		cv::Mat roughCut;//粗提取用的mask

		std::vector<_KEY_SEGMENT> result;//存储结果

		//实验性数据存储
		cv::Mat imgMeanshift;//采用meanshift算法

	public:

		//读取图像
		void loadImg(cv::Mat srcimage)
		{
			srcimage.copyTo(imgOriginal);
		}




		//生成用于粗提取的mask
		void buildMask(cv::Mat imgMask0)
		{
			
			cv::Mat mask0(imgMask0.size(), CV_8UC1);
			cv::Mat mask1(imgMask0.size(), CV_8UC1);
			//for (int i = 0; i < imgMask0.rows; i++)//行数(会有神奇的事情发生，不知为何)
			//{
			//	for (int j = 0; j < imgMask0.cols; j++)
			//	{
			//		if (imgMask0.at<uchar>(i, j) < 10)
			//		{
			//			mask0.at<uchar>(i, j) = cv::GC_FGD;
			//		}
			//		//if (10<=imgMask0.at<uchar>(i, j) <=120)
			//		//{
			//		//	//cv::Rect r1(i, j, 1, 1);
			//		//	//mask(r1).setTo((unsigned int)3);
			//		//	mask0.at<uchar>(i, j) = cv::GC_PR_FGD;
			//		//}
			//		//if (imgMask0.at<uchar>(i, j)>120)
			//		//{
			//		//	mask0.at<uchar>(i, j) = cv::GC_BGD;
			//		//}
			//		else { mask0.at<uchar>(i, j) = 0; }
			//		//std::cout << j << std::endl;
			//	}
			//}
			cv::Mat maskTem1,maskTem3;//存储 前景  可能前景
			cv::compare(imgMask0, 10, maskTem1, cv::CMP_LT);
			cv::compare(imgMask0, 250, maskTem3, cv::CMP_LT);
			maskTem3 = maskTem3 - maskTem1;//可能前景+前景减去前景=可能前景
			maskTem1 = maskTem1 / 255;
			maskTem3 = maskTem3 / 255;
			maskTem3 = maskTem3 * 3;//可能前景用3表示
			mask0 = maskTem1 + maskTem3;
			//mask0.convertTo(mask0, CV_8UC1);

			cv::flip(mask0, mask1,1);//按照y轴镜像得到mask1

			cv::Mat test;
			imgOriginal.copyTo(test, mask0);
			cv::imwrite("Data\\test.jpg", test);

			//cv::imshow("test", imgMask0);
			//std::cout << mask0.type() << std::endl;
			cv::FileStorage fs0("Data\\mask0.xml", cv::FileStorage::WRITE);
			fs0 << "mask0" << mask0;
			fs0.release();
			cv::FileStorage fs1("Data\\mask1.xml", cv::FileStorage::WRITE);
			fs1 << "mask1" << mask1;
			fs1.release();
		}


		//给出结果图像
		cv::Mat getimgresult()
		{
			cv::Mat imgResult;
			imgOriginal.copyTo(imgResult);
			int unsure = 0;
			for (int i = 0; i < result.size(); i++)
			{
				if (result[i].key == 0)
				{
					cv::circle(imgResult, result[i].ep0, 3, cv::Scalar(255, 0, 0), 2);
					cv::circle(imgResult, result[i].ep1, 3, cv::Scalar(255, 0, 0), 2);
				}
				if (result[i].key == 1)
				{
					cv::circle(imgResult, result[i].ep0, 3, cv::Scalar(0, 0, 255), 2);
					cv::circle(imgResult, result[i].ep1, 3, cv::Scalar(0, 0, 255), 2);
				}
				if (result[i].key == -1)
				{
					cv::circle(imgResult, result[i].ep0, 3, cv::Scalar(0, 255, 255), 2);
					cv::circle(imgResult, result[i].ep1, 3, cv::Scalar(0, 255, 255), 2);
					unsure++;
				}
			}
			return imgResult;
		}



		//图像对比度增强
		void imgAdjust()
		{

			//===============这种办法不好用===================//
			//cv::Mat mergeImg;//合并后的图像
			////用来存储各通道图片的向量
			//std::vector<cv::Mat> splitBGR(imgOriginal.channels());
			////分割通道，存储到splitBGR中
			//split(imgOriginal, splitBGR);
			////对各个通道分别进行直方图均衡化
			//for (int i = 0; i<imgOriginal.channels(); i++)
			//	equalizeHist(splitBGR[i], splitBGR[i]);
			////合并通道
			//merge(splitBGR, mergeImg);

			//mergeImg.copyTo(imgSharpen);//存储锐化后的图像




			imgOriginal.copyTo(imgSharpen);
		}
		//存储结果
		void saveResult(std::vector<_KEY_SEGMENT>&segments)
		{
			result.assign(segments.begin(), segments.end());
		}

		//确定mask位置（不是很好用，没用上）
		void setMask()
		{
			cv::Mat maskTem0;
			cv::Mat maskTem1;
			cv::Mat afMask0(imgOriginal.size(), imgOriginal.type());
			cv::Mat afMask1(imgOriginal.size(), imgOriginal.type());

			cv::FileStorage fs0("Data\\mask0.xml", cv::FileStorage::READ);
			fs0["mask0"] >> maskTem0;
			cv::compare(maskTem0, cv::GC_PR_FGD, maskTem0, cv::CMP_EQ);

			cv::FileStorage fs1("Data\\mask1.xml", cv::FileStorage::READ);
			fs1["mask1"] >> maskTem1;
			cv::compare(maskTem1, cv::GC_PR_FGD, maskTem1, cv::CMP_EQ);

		/*	std::cout << maskTem0.size()<<std::endl;*/

			imgOriginal.copyTo(afMask0, maskTem0);
			imgOriginal.copyTo(afMask0, maskTem1);

			cv::Scalar s0 = cv::mean(afMask0);
			cv::Scalar s1 = cv::mean(afMask1);

			if (s0.val[1] > s1.val[1]){ f_or_b = 0; }
			else { f_or_b = 1; }
		}


		//粗提取图像
		void grabCut(bool f_or_b)
		{
			FindSegmentMarker::f_or_b = f_or_b;
			cv::Mat mask(imgSharpen.size(),CV_8UC1);
			if (f_or_b == 0)
			{
				cv::FileStorage fs("Data\\mask0.xml", cv::FileStorage::READ);
				fs["mask0"] >> mask;
			}
			else
			{
				cv::FileStorage fs("Data\\mask1.xml", cv::FileStorage::READ);
				fs["mask1"] >> mask;
			}
			
			
			cv::Mat bgModel, fgModel;//;中间变量
			cv::Mat result;//mask，中间变量

			cv::grabCut(imgSharpen, mask, rectangle, bgModel, fgModel, 2, cv::GC_INIT_WITH_MASK);//关键函数,划分大致区域

			//比较函数保留值为GC_PR_FGD的像素(改为留下为前景的部分)
			cv::Mat maskTem1,maskTem3;
			cv::compare(mask, cv::GC_FGD, maskTem1, cv::CMP_EQ);
			cv::compare(mask, cv::GC_PR_FGD, maskTem3, cv::CMP_EQ);
			mask = maskTem1 + maskTem3;
			// ----------------------------------产生输出图像----------------------------//
			cv::Mat Foreground(imgSharpen.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			//背景值为 GC_BGD=0，作为掩码
			imgSharpen.copyTo(Foreground, mask);
			//cv::imwrite("Data\\afterMask.jpg", Foreground);
			Foreground.copyTo(imgForeground);

			cv::imwrite("Data\\fore.jpg", imgForeground);

			//以前使用的划分长方形区域的方法，已经舍弃
			//cv::Mat bgModel, fgModel;//;中间变量
			//cv::Mat result;//mask，中间变量
			//rectangle = (f_or_b == 0) ? cv::Rect(60, 650, 220, 560) : cv::Rect(800, 311, 210, 1318);
			//cv::grabCut(imgOriginal, result, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);//关键函数,划分大致区域
			////比较函数保留值为GC_PR_FGD的像素
			//cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
			//// 产生输出图像
			//cv::Mat Foreground(imgOriginal.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			////背景值为 GC_BGD=0，作为掩码
			//imgOriginal.copyTo(Foreground, result);
			//Foreground.copyTo(imgForeground);//粗提取完成
		}



		////设置粗提取所用到的mask
		//void setMask(cv::Mat)
		//{
		//	
		//}


		////重载,用mask方式粗提取
		//void grab(cv::Mat roughCut)
		//{
		//	
		//}

		//粗提取后图像二值化
		void imgThreshold(double lightThreshold=60)//默认为60
		{
			cv::Mat foreGray(imgOriginal.size(),CV_32FC1);
			cv::Mat foreBinary(imgOriginal.size(), CV_32FC1);

			//假如不做灰度值折算而是拿出绿色通道作为“灰度图”？(通过，效果还可以)
			std::vector<cv::Mat> BGR;
			cv::split(imgForeground, BGR);
			BGR[1].copyTo(foreGray);
			//cv::cvtColor(imgForeground, foreGray, CV_BGR2GRAY);//转化为灰度图像，方便转为二值图

			//假如将得到的灰度图像也再锐化一遍会不会降低阈值设置的难度(不会，图像变成大片白色，仔细看看锐化的原理再用吧)
			/*cv::equalizeHist(foreGray, foreGray);*/

			cv::threshold(foreGray, foreBinary, 85, 255, 1);//得到只有线段轮廓的二值图像（坑2，应该根据图像总体亮度进行判断。仍然可以按照kmeans方法或者cvAvg函数）

	
			foreGray.copyTo(imgForGray);
			foreBinary.copyTo(imgBinary);
			cv::imwrite("Data\\gray.jpg", imgForGray);
			cv::imwrite("Data\\Binary.jpg", imgBinary);
		}


		//查找轮廓并根据多种方法剔除无效轮廓
		void findContours()
		{
			//从二值图像里找
			cv::findContours(imgBinary,
				roughContours, // a vector of contours 
				CV_RETR_EXTERNAL, // retrieve the external contours
				CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

			//假如说从灰度图像里找会不会更靠谱？(不会，根本搜索不到)
			//cv::findContours(imgForGray,
			//	roughContours, // a vector of contours 
			//	CV_RETR_EXTERNAL, // retrieve the external contours
			//	CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

			//先剔除值过大或者过小的部分（坑3，剔除除了直线之外的轮廓）
			for (int i = 0; i < roughContours.size(); i++)
			{

				std::pair<cv::Point, cv::Point> f_s;
				f_s = detExtrePoi(roughContours[i]);


				cv::Point dreamCenter = (f_or_b) ? cv::Point(100, 960) : cv::Point(980,960);
				double alpha = atan(fabs((double)(f_s.first.y - dreamCenter.y) / (f_s.first.x - dreamCenter.x)));
				double beta = atan(fabs((double)(f_s.first.y - f_s.second.y) / (f_s.first.x - f_s.second.x)));//坑5，角度的判断应该和线段位置有关

				double theta = alpha - beta;
				cv::Point poi0Tem = detExtrePoi(roughContours[i]).first;//点阵最外
				cv::Point poi1Tem = detExtrePoi(roughContours[i]).second;//点阵最内
				double dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
				//std::cout << "轮廓大小：" << roughContours[i].size() << std::endl;
				//std::cout << "线段长度：" << dis_line << std::endl;
				//std::cout << "倾斜角:" << theta << std::endl;
				if ((roughContours[i].size()>15) && (roughContours[i].size() < 500) && (theta<CV_PI / 9) && (dis_line>12))//注意灯太亮时对周围的点有很大影响
				{
					exactContours.push_back(roughContours[i]);//到此得到较为准确的轮廓组
				}
			}
			//然后按照Y值从小到大排序
			for (int i = 0; i < exactContours.size(); i++)
			{
				for (int j = i + 1; j < exactContours.size(); j++)
				{
					int yI, yJ;
					yI = detExtrePoi(exactContours[i]).first.y;
					yJ = detExtrePoi(exactContours[j]).first.y;
					if (yI > yJ)
					{
						swap(exactContours[i], exactContours[j]);
					}
				}
			}
			std::cout << "检测轮廓数量：" << exactContours.size() << std::endl;
		}
		


		//查找轮廓两端的点(步骤中间用到)
		std::pair<cv::Point, cv::Point> detExtrePoi(std::vector<cv::Point> Points)
		{
			cv::Point poiYMax, poiYMin;
			poiYMax = poiYMin = Points[0];
			for (int i = 0; i < Points.size(); i++)
			{
				if (Points[i].x > poiYMax.x)
				{
					poiYMax = Points[i];
				}
				if (Points[i].x <poiYMin.x)
				{
					poiYMin = Points[i];
				}
			}
			return std::make_pair(poiYMin, poiYMax);//端点顺序都是由图像外到内
		}


		//判断线段长度
		char judgeDis(cv::Point poi1, cv::Point poi2, double sSize, double lSize)
		{
			char k;
			double dis = sqrt(powf((poi1.x - poi2.x), 2) + powf((poi1.y - poi2.y), 2));
			if (dis > lSize){ k = 1; }//（坑4）阈值可以考虑采用聚类算法进行分析
			else if (dis < sSize){ k = 0; }
			else { k = -1; }
			return k;
		}

		char judgeDis(cv::Point poi1, cv::Point poi2,int num)
		{
			double lSize, sSize;
			lSize = shortTV[num/8];
			sSize = longTV[num/8];
			char k;
			double dis = sqrt(powf((poi1.x - poi2.x), 2) + powf((poi1.y - poi2.y), 2));
			if (dis > lSize){ k = 1; }//（坑4）阈值可以考虑采用聚类算法进行分析
			else if (dis < sSize){ k = 0; }
			else { k = -1; }
			return k;
		}


		//设置线段长短的分界阈值
		void setLSThreshold()
		{
			
			cv::Mat lineDiss(exactContours.size(), 1, CV_32FC1);//用于存储每条线段距离
			for (int i = 0; i < exactContours.size(); i++)
			{
				cv::Point poi0Tem = detExtrePoi(exactContours[i]).first;//点阵最外
				cv::Point poi1Tem = detExtrePoi(exactContours[i]).second;//点阵最内
				float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
				lineDiss.at<float>(i) = dis_line;
			}

			std::vector<double> center;//存储聚类中心
			cv::Mat k;//存储标签
			cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1.0);
			cv::kmeans(lineDiss, 2, k, criteria, 10, 0, center);
			shortThreshold = center[0] + (center[1] - center[0]) / 3;
			longThreshold = center[1] - (center[1] - center[0]) / 3;//等分为三段


			//采用分段取阈值的方法

			int group = exactContours.size() / 8;//取周围一共八条直线作为判断的标准（前面的比较密，取得多一些）
			int remainder = exactContours.size() % 8;
			for (int i = 0; i < group; i++)
			{
				double aveDis=0;
				/*cv::Mat lineDissGroup(6, 1, CV_32FC1);*/
				for (int count = 1; count <= 8; count++)
				{
					cv::Point poi0Tem = detExtrePoi(exactContours[i * 8 + count]).first;//点阵最外
					cv::Point poi1Tem = detExtrePoi(exactContours[i * 8 + count]).second;//点阵最内
					float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
					aveDis += dis_line;
					//lineDissGroup.at<float>(i) = dis_line;
				}
				aveDis /= 8;
				//std::vector<double> center;//存储聚类中心
				//cv::Mat k;//存储标签
				//cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1.0);
				//cv::kmeans(lineDissGroup, 2, k, criteria, 10, 0, center);
				shortThreshold = aveDis*0.9;
				longThreshold = aveDis*1.1;

				shortTV.push_back(shortThreshold);
				longTV.push_back(longThreshold);
			}

			
			for (int num = 0; num < 1; num++)//纯粹为了保持一致，只运行一遍
			{
				double aveDis=0;
				//cv::Mat lineDissGroup(6, 1, CV_32FC1);
				for (int i = exactContours.size()-1; i > exactContours.size() - 7; i--)
				{
					cv::Point poi0Tem = detExtrePoi(exactContours[i]).first;//点阵最外
					cv::Point poi1Tem = detExtrePoi(exactContours[i]).second;//点阵最内
					float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
					aveDis += dis_line;
					/*lineDissGroup.at<float>(i) = dis_line;*/
				}
				aveDis /= 6;
				//std::vector<double> center;//存储聚类中心
				//cv::Mat k;//存储标签
				//cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1.0);
				//cv::kmeans(lineDissGroup, 2, k, criteria, 10, 0, center);
				shortThreshold = aveDis*0.9;
				longThreshold = aveDis*1.1;

				shortTV.push_back(shortThreshold);
				longTV.push_back(longThreshold);
			}



		}
		void setLightThreshold()
		{
			
		}


		//===============================================实验性方法====================================================//
		//尝试meanshift方法粗提取
		cv::Mat  testMeanhift()
		{
			cv::pyrMeanShiftFiltering(imgOriginal, imgMeanshift, 10, 10, 1);
			return imgMeanshift;
		}
		//尝试从色彩通道采集图像
		cv::Mat testColor()
		{
			cv::Mat colorSplit(imgOriginal.size(), CV_32FC1);
			std::vector<cv::Mat> channels;
			cv::split(imgOriginal, channels);

			for (int i = 0; i < imgOriginal.cols; i++)
			{
				for (int j = 0; j < imgOriginal.rows; j++)
				{
					uchar gb, gr;
					gb = channels[1].at<uchar>(i, j) - channels[0].at<uchar>(i, j);
					gr = channels[1].at<uchar>(i, j) - channels[2].at<uchar>(i, j);
					if (gr > 30 && gb > 30)
					{
						colorSplit.at<float>(i, j) = 250;
					}
					else
					{
						colorSplit.at<float>(i, j) = 0;
					}
				}
			}

			return colorSplit;
		}
		//用mask的方法提取图像
//		cv::Mat testMask()//(已经集成
//
//		{
//			cv::FileStorage fs("Data\\mask0.xml", cv::FileStorage::READ);
//			cv::Mat mask;
//			fs["mask0"] >> mask;
//			cv::Mat bgModel, fgModel;//;中间变量
//			cv::Mat result;//mask，中间变量
//
//			cv::grabCut(imgOriginal, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_MASK);//关键函数,划分大致区域
//			//比较函数保留值为GC_PR_FGD的像素
//			cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
//			// 产生输出图像
//			cv::Mat Foreground(imgOriginal.size(), CV_8UC3, cv::Scalar(255, 255, 255));
//			//背景值为 GC_BGD=0，作为掩码
//			imgOriginal.copyTo(Foreground, mask);
//			cv::imwrite("Data\\afterMask.jpg", Foreground);
//			return Foreground;
//		}










};
