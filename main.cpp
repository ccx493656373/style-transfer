																																 //核心代码： 加载模型 -> 读取图片 -> 前传计算 -> 输出图片
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

	using namespace cv;
	using namespace std;

	//----------------cv中取出	-----------------开始
	const int CV_MAX_DIM = 32;
	Mat getPlane(const Mat &m, int n, int cn)
	{
		CV_Assert(m.dims > 2);
		int sz[CV_MAX_DIM];
		for (int i = 2; i < m.dims; i++)
		{
			sz[i - 2] = m.size.p[i];
		}
		return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
	}

	//用于单图，如果多图还要再修改
	void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
	{
		//blob 是浮点精度的4维矩阵
		//blob_[0] = 批量大小 = 图像数
		//blob_[1] = 通道数
		//blob_[2] = 高度
		//blob_[3] = 宽度    
		CV_Assert(blob_.depth() == CV_32F);
		CV_Assert(blob_.dims == 4);

		//images_.create(cv::Size(1, blob_.size[0]),blob_.depth() );//多图，不明白为什么？
		images_.create(blob_.size[2], blob_.size[3], blob_.depth());//创建一个图像


		std::vector<Mat> vectorOfChannels(blob_.size[1]);
		//for (int n = 0; n <  blob_.size[0]; ++n) //多个图
		{int n = 0;                                //只有一个图
		for (int c = 0; c < blob_.size[1]; ++c)
		{
			vectorOfChannels[c] = getPlane(blob_, n, c);
		}
		//cv::merge(vectorOfChannels, images_.getMatRef(n));//这里会出错，是前面的create的原因？
		cv::merge(vectorOfChannels, images_);//通道合并
		}
	}
	//----------------cv中取出	-----------------结束

	int main(int argc, char *argv[])
	{
		char jpgname[256];//图像名
		if (argc == 2)
			strcpy_s(jpgname, argv[1]);
		else
			strcpy_s(jpgname, "C:/Users/asus/Desktop/in/四谎.jpg");

		double time1 = static_cast<double>(getTickCount());  //记录起始时间
															 // 加载模型
		dnn::Net net = cv::dnn::readNetFromTorch("C:/Users/asus/Desktop/XilinxSUMMER/Project/Neural-Style-Transfer-master/Transfer/candy.t7");

		// 读取图片
		Mat image = cv::imread(jpgname);
		size_t h = image.rows;// 行数(高度)
		size_t w = image.cols;// 列数（宽度）



		Mat inputBlob;//转换为 (1,3,h,w) 的矩阵 即：(图像数,通道,高,宽)
					  //blobFromImage函数解释:

					  //第一个参数，InputArray image，表示输入的图像，可以是opencv的mat数据类型。
					  //第二个参数，scalefactor，这个参数很重要的，如果训练时，是归一化到0-1之间，那么这个参数就应该为0.00390625f （1/256），否则为1.0
					  //第三个参数，size，应该与训练时的输入图像尺寸保持一致。
					  //第四个参数，mean，这个主要在caffe中用到，caffe中经常会用到训练数据的均值。tf中貌似没有用到均值文件。
					  //第五个参数，swapRB，是否交换图像第1个通道和最后一个通道的顺序。
					  //第六个参数，crop，如果为true，就是裁剪图像，如果为false，就是等比例放缩图像。
					  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416), Scalar(), false, true);//1/255.F
					  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416*h/w), Scalar(103.939, 116.779, 123.680), false, true);//
		inputBlob = cv::dnn::blobFromImage(image, 1.0, Size(600,400), Scalar(103.939, 116.779, 123.680), false, false);//
																		// 进行计算
		net.setInput(inputBlob);
		Mat out = net.forward();

		Mat Styled;//4维转回3维
				   //cv::dnn::imagesFromBlob(out, Styled);//由于这个函数会出错，已把它从dnn取出稍稍修改一下用
		imagesFromBlob(out, Styled);
		// 输出图片
		Styled /= 255;

		Mat uStyled;
		Styled.convertTo(uStyled, CV_8U, 255);//转换格式
		cv::imshow("风格图像", uStyled);
		cv::imwrite("风格图像.jpg", uStyled);

		waitKey(0);
		return 0;
	}