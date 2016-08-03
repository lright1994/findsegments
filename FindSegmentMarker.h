
#pragma once

#include "opencv2/opencv.hpp" 
#include <math.h>
#include "PanoStitcher.h"
#include "math.h"
#include <iostream>

typedef struct
{
	char key;		//�߶α��룺��0����1����-1��
	cv::Point ep0;	//�߶ζ˵�0����(��������)
	cv::Point ep1;	//�߶ζ˵�1���꣨�������꣩
	//cv::Point pi0;	//�߶ζ˵�0���꣨�������꣩
	//cv::Point pi1;	//�߶ζ˵�1���꣨�������꣩
}_KEY_SEGMENT;

#define minv(x,y) ( (x>y)?y:x)

class FindSegmentMarker
{
	public:
	//--------------------------------------------------------------���캯��--------------------------------------------------------//

	FindSegmentMarker(cv::Mat srcimage) //��Ӧ����ͼ��������Լ��������012, ����0���������ң�1��
	{
		loadImg(srcimage);//����ͼ��
		//setMask();//����mask(��Ϊ��̫׼ȷ���Ȳ���)

	}
	//--------------------------------------------------------------����ӿ�---------------------------------------------------------//
	void getResult(std::vector<_KEY_SEGMENT>&segments, bool f_or_b)//��������Ľӿ�
	{
		imgAdjust();//��ǿͼ��Աȶ�

		grabCut(f_or_b);//����ȡ
		imgThreshold();//��ֵ��
		findContours();//�ҵ�����
		setLSThreshold();//���ó�����ֵ


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
	//===============================================================������ȡ���ݵĺ���=======================================================//
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

		//===================================================��Ҫ�����洢�����м����ݱ��ڵ���=================================================//
		bool f_or_b;//ȷ������or������

		cv::Mat imgOriginal;//ԭʼͼ��
		cv::Mat imgForeground;//����ȡ���ͼ��
		cv::Mat imgForGray;//����ȡ��תΪ�Ҷ�ͼ��
		cv::Mat imgBinary;//����ȡ���ͼ����ж�ֵ��
		cv::Mat imgSort;//���������ͼƬ���ɴӺ�ɫ����Ƿ񽥱��ж�
		cv::Mat imgSharpen;//��ͼƬ�����ӶԱȶ�

		cv::Mat lineDiss;//�����洢ÿ��ֱ�߳��ȵ�һά����(validCon.size(), 1, CV_32FC1)
		cv::Mat mask;//�洢����ȡ��Ҫ��mask

		std::vector<std::vector<cv::Point>> roughContours;//����ȡ����
		std::vector<std::vector<cv::Point>> exactContours;//�޳���Ч���������Ч����
		
		double shortThreshold, longThreshold;//�ж�ֱ�߳��ȵ�������ֵ
		std::vector<double> shortTV, longTV;
		double lightThreshold;//ͼ���ֵ������ֵ

		cv::Rect rectangle;//����ȡʱ�����򣬿�1������mask����
		cv::Mat roughCut;//����ȡ�õ�mask

		std::vector<_KEY_SEGMENT> result;//�洢���

		//ʵ�������ݴ洢
		cv::Mat imgMeanshift;//����meanshift�㷨

	public:

		//��ȡͼ��
		void loadImg(cv::Mat srcimage)
		{
			srcimage.copyTo(imgOriginal);
		}




		//�������ڴ���ȡ��mask
		void buildMask(cv::Mat imgMask0)
		{
			
			cv::Mat mask0(imgMask0.size(), CV_8UC1);
			cv::Mat mask1(imgMask0.size(), CV_8UC1);
			//for (int i = 0; i < imgMask0.rows; i++)//����(������������鷢������֪Ϊ��)
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
			cv::Mat maskTem1,maskTem3;//�洢 ǰ��  ����ǰ��
			cv::compare(imgMask0, 10, maskTem1, cv::CMP_LT);
			cv::compare(imgMask0, 250, maskTem3, cv::CMP_LT);
			maskTem3 = maskTem3 - maskTem1;//����ǰ��+ǰ����ȥǰ��=����ǰ��
			maskTem1 = maskTem1 / 255;
			maskTem3 = maskTem3 / 255;
			maskTem3 = maskTem3 * 3;//����ǰ����3��ʾ
			mask0 = maskTem1 + maskTem3;
			//mask0.convertTo(mask0, CV_8UC1);

			cv::flip(mask0, mask1,1);//����y�᾵��õ�mask1

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


		//�������ͼ��
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



		//ͼ��Աȶ���ǿ
		void imgAdjust()
		{

			//===============���ְ취������===================//
			//cv::Mat mergeImg;//�ϲ����ͼ��
			////�����洢��ͨ��ͼƬ������
			//std::vector<cv::Mat> splitBGR(imgOriginal.channels());
			////�ָ�ͨ�����洢��splitBGR��
			//split(imgOriginal, splitBGR);
			////�Ը���ͨ���ֱ����ֱ��ͼ���⻯
			//for (int i = 0; i<imgOriginal.channels(); i++)
			//	equalizeHist(splitBGR[i], splitBGR[i]);
			////�ϲ�ͨ��
			//merge(splitBGR, mergeImg);

			//mergeImg.copyTo(imgSharpen);//�洢�񻯺��ͼ��




			imgOriginal.copyTo(imgSharpen);
		}
		//�洢���
		void saveResult(std::vector<_KEY_SEGMENT>&segments)
		{
			result.assign(segments.begin(), segments.end());
		}

		//ȷ��maskλ�ã����Ǻܺ��ã�û���ϣ�
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


		//����ȡͼ��
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
			
			
			cv::Mat bgModel, fgModel;//;�м����
			cv::Mat result;//mask���м����

			cv::grabCut(imgSharpen, mask, rectangle, bgModel, fgModel, 2, cv::GC_INIT_WITH_MASK);//�ؼ�����,���ִ�������

			//�ȽϺ�������ֵΪGC_PR_FGD������(��Ϊ����Ϊǰ���Ĳ���)
			cv::Mat maskTem1,maskTem3;
			cv::compare(mask, cv::GC_FGD, maskTem1, cv::CMP_EQ);
			cv::compare(mask, cv::GC_PR_FGD, maskTem3, cv::CMP_EQ);
			mask = maskTem1 + maskTem3;
			// ----------------------------------�������ͼ��----------------------------//
			cv::Mat Foreground(imgSharpen.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			//����ֵΪ GC_BGD=0����Ϊ����
			imgSharpen.copyTo(Foreground, mask);
			//cv::imwrite("Data\\afterMask.jpg", Foreground);
			Foreground.copyTo(imgForeground);

			cv::imwrite("Data\\fore.jpg", imgForeground);

			//��ǰʹ�õĻ��ֳ���������ķ������Ѿ�����
			//cv::Mat bgModel, fgModel;//;�м����
			//cv::Mat result;//mask���м����
			//rectangle = (f_or_b == 0) ? cv::Rect(60, 650, 220, 560) : cv::Rect(800, 311, 210, 1318);
			//cv::grabCut(imgOriginal, result, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);//�ؼ�����,���ִ�������
			////�ȽϺ�������ֵΪGC_PR_FGD������
			//cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
			//// �������ͼ��
			//cv::Mat Foreground(imgOriginal.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			////����ֵΪ GC_BGD=0����Ϊ����
			//imgOriginal.copyTo(Foreground, result);
			//Foreground.copyTo(imgForeground);//����ȡ���
		}



		////���ô���ȡ���õ���mask
		//void setMask(cv::Mat)
		//{
		//	
		//}


		////����,��mask��ʽ����ȡ
		//void grab(cv::Mat roughCut)
		//{
		//	
		//}

		//����ȡ��ͼ���ֵ��
		void imgThreshold(double lightThreshold=60)//Ĭ��Ϊ60
		{
			cv::Mat foreGray(imgOriginal.size(),CV_32FC1);
			cv::Mat foreBinary(imgOriginal.size(), CV_32FC1);

			//���粻���Ҷ�ֵ��������ó���ɫͨ����Ϊ���Ҷ�ͼ����(ͨ����Ч��������)
			std::vector<cv::Mat> BGR;
			cv::split(imgForeground, BGR);
			BGR[1].copyTo(foreGray);
			//cv::cvtColor(imgForeground, foreGray, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ�񣬷���תΪ��ֵͼ

			//���罫�õ��ĻҶ�ͼ��Ҳ����һ��᲻�ή����ֵ���õ��Ѷ�(���ᣬͼ���ɴ�Ƭ��ɫ����ϸ�����񻯵�ԭ�����ð�)
			/*cv::equalizeHist(foreGray, foreGray);*/

			cv::threshold(foreGray, foreBinary, 85, 255, 1);//�õ�ֻ���߶������Ķ�ֵͼ�񣨿�2��Ӧ�ø���ͼ���������Ƚ����жϡ���Ȼ���԰���kmeans��������cvAvg������

	
			foreGray.copyTo(imgForGray);
			foreBinary.copyTo(imgBinary);
			cv::imwrite("Data\\gray.jpg", imgForGray);
			cv::imwrite("Data\\Binary.jpg", imgBinary);
		}


		//�������������ݶ��ַ����޳���Ч����
		void findContours()
		{
			//�Ӷ�ֵͼ������
			cv::findContours(imgBinary,
				roughContours, // a vector of contours 
				CV_RETR_EXTERNAL, // retrieve the external contours
				CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

			//����˵�ӻҶ�ͼ�����һ᲻������ף�(���ᣬ������������)
			//cv::findContours(imgForGray,
			//	roughContours, // a vector of contours 
			//	CV_RETR_EXTERNAL, // retrieve the external contours
			//	CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

			//���޳�ֵ������߹�С�Ĳ��֣���3���޳�����ֱ��֮���������
			for (int i = 0; i < roughContours.size(); i++)
			{

				std::pair<cv::Point, cv::Point> f_s;
				f_s = detExtrePoi(roughContours[i]);


				cv::Point dreamCenter = (f_or_b) ? cv::Point(100, 960) : cv::Point(980,960);
				double alpha = atan(fabs((double)(f_s.first.y - dreamCenter.y) / (f_s.first.x - dreamCenter.x)));
				double beta = atan(fabs((double)(f_s.first.y - f_s.second.y) / (f_s.first.x - f_s.second.x)));//��5���Ƕȵ��ж�Ӧ�ú��߶�λ���й�

				double theta = alpha - beta;
				cv::Point poi0Tem = detExtrePoi(roughContours[i]).first;//��������
				cv::Point poi1Tem = detExtrePoi(roughContours[i]).second;//��������
				double dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
				//std::cout << "������С��" << roughContours[i].size() << std::endl;
				//std::cout << "�߶γ��ȣ�" << dis_line << std::endl;
				//std::cout << "��б��:" << theta << std::endl;
				if ((roughContours[i].size()>15) && (roughContours[i].size() < 500) && (theta<CV_PI / 9) && (dis_line>12))//ע���̫��ʱ����Χ�ĵ��кܴ�Ӱ��
				{
					exactContours.push_back(roughContours[i]);//���˵õ���Ϊ׼ȷ��������
				}
			}
			//Ȼ����Yֵ��С��������
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
			std::cout << "�������������" << exactContours.size() << std::endl;
		}
		


		//�����������˵ĵ�(�����м��õ�)
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
			return std::make_pair(poiYMin, poiYMax);//�˵�˳������ͼ���⵽��
		}


		//�ж��߶γ���
		char judgeDis(cv::Point poi1, cv::Point poi2, double sSize, double lSize)
		{
			char k;
			double dis = sqrt(powf((poi1.x - poi2.x), 2) + powf((poi1.y - poi2.y), 2));
			if (dis > lSize){ k = 1; }//����4����ֵ���Կ��ǲ��þ����㷨���з���
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
			if (dis > lSize){ k = 1; }//����4����ֵ���Կ��ǲ��þ����㷨���з���
			else if (dis < sSize){ k = 0; }
			else { k = -1; }
			return k;
		}


		//�����߶γ��̵ķֽ���ֵ
		void setLSThreshold()
		{
			
			cv::Mat lineDiss(exactContours.size(), 1, CV_32FC1);//���ڴ洢ÿ���߶ξ���
			for (int i = 0; i < exactContours.size(); i++)
			{
				cv::Point poi0Tem = detExtrePoi(exactContours[i]).first;//��������
				cv::Point poi1Tem = detExtrePoi(exactContours[i]).second;//��������
				float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
				lineDiss.at<float>(i) = dis_line;
			}

			std::vector<double> center;//�洢��������
			cv::Mat k;//�洢��ǩ
			cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1.0);
			cv::kmeans(lineDiss, 2, k, criteria, 10, 0, center);
			shortThreshold = center[0] + (center[1] - center[0]) / 3;
			longThreshold = center[1] - (center[1] - center[0]) / 3;//�ȷ�Ϊ����


			//���÷ֶ�ȡ��ֵ�ķ���

			int group = exactContours.size() / 8;//ȡ��Χһ������ֱ����Ϊ�жϵı�׼��ǰ��ıȽ��ܣ�ȡ�ö�һЩ��
			int remainder = exactContours.size() % 8;
			for (int i = 0; i < group; i++)
			{
				double aveDis=0;
				/*cv::Mat lineDissGroup(6, 1, CV_32FC1);*/
				for (int count = 1; count <= 8; count++)
				{
					cv::Point poi0Tem = detExtrePoi(exactContours[i * 8 + count]).first;//��������
					cv::Point poi1Tem = detExtrePoi(exactContours[i * 8 + count]).second;//��������
					float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
					aveDis += dis_line;
					//lineDissGroup.at<float>(i) = dis_line;
				}
				aveDis /= 8;
				//std::vector<double> center;//�洢��������
				//cv::Mat k;//�洢��ǩ
				//cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1.0);
				//cv::kmeans(lineDissGroup, 2, k, criteria, 10, 0, center);
				shortThreshold = aveDis*0.9;
				longThreshold = aveDis*1.1;

				shortTV.push_back(shortThreshold);
				longTV.push_back(longThreshold);
			}

			
			for (int num = 0; num < 1; num++)//����Ϊ�˱���һ�£�ֻ����һ��
			{
				double aveDis=0;
				//cv::Mat lineDissGroup(6, 1, CV_32FC1);
				for (int i = exactContours.size()-1; i > exactContours.size() - 7; i--)
				{
					cv::Point poi0Tem = detExtrePoi(exactContours[i]).first;//��������
					cv::Point poi1Tem = detExtrePoi(exactContours[i]).second;//��������
					float dis_line = sqrt(powf((poi0Tem.x - poi1Tem.x), 2) + powf((poi1Tem.y - poi0Tem.y), 2));
					aveDis += dis_line;
					/*lineDissGroup.at<float>(i) = dis_line;*/
				}
				aveDis /= 6;
				//std::vector<double> center;//�洢��������
				//cv::Mat k;//�洢��ǩ
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


		//===============================================ʵ���Է���====================================================//
		//����meanshift��������ȡ
		cv::Mat  testMeanhift()
		{
			cv::pyrMeanShiftFiltering(imgOriginal, imgMeanshift, 10, 10, 1);
			return imgMeanshift;
		}
		//���Դ�ɫ��ͨ���ɼ�ͼ��
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
		//��mask�ķ�����ȡͼ��
//		cv::Mat testMask()//(�Ѿ�����
//
//		{
//			cv::FileStorage fs("Data\\mask0.xml", cv::FileStorage::READ);
//			cv::Mat mask;
//			fs["mask0"] >> mask;
//			cv::Mat bgModel, fgModel;//;�м����
//			cv::Mat result;//mask���м����
//
//			cv::grabCut(imgOriginal, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_MASK);//�ؼ�����,���ִ�������
//			//�ȽϺ�������ֵΪGC_PR_FGD������
//			cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
//			// �������ͼ��
//			cv::Mat Foreground(imgOriginal.size(), CV_8UC3, cv::Scalar(255, 255, 255));
//			//����ֵΪ GC_BGD=0����Ϊ����
//			imgOriginal.copyTo(Foreground, mask);
//			cv::imwrite("Data\\afterMask.jpg", Foreground);
//			return Foreground;
//		}










};