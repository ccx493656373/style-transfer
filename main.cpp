																																 //���Ĵ��룺 ����ģ�� -> ��ȡͼƬ -> ǰ������ -> ���ͼƬ
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

	using namespace cv;
	using namespace std;


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

	void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
	{

		CV_Assert(blob_.depth() == CV_32F);
		CV_Assert(blob_.dims == 4);


		images_.create(blob_.size[2], blob_.size[3], blob_.depth());//����һ��ͼ��


		std::vector<Mat> vectorOfChannels(blob_.size[1]);

		{int n = 0;             
		for (int c = 0; c < blob_.size[1]; ++c)
		{
			vectorOfChannels[c] = getPlane(blob_, n, c);
		}

		cv::merge(vectorOfChannels, images_);//ͨ���ϲ�
		}
	}
	//----------------cv��ȡ��	-----------------����

	int main(int argc, char *argv[])
	{
		char jpgname[256];//ͼ����
		if (argc == 2)
			strcpy_s(jpgname, argv[1]);
		else
			strcpy_s(jpgname, "C:/Users/asus/Desktop/in/�Ļ�.jpg");

		double time1 = static_cast<double>(getTickCount());  //��¼��ʼʱ��
															 // ����ģ��
		dnn::Net net = cv::dnn::readNetFromTorch("C:/Users/asus/Desktop/XilinxSUMMER/Project/Neural-Style-Transfer-master/Transfer/candy.t7");

		// ��ȡͼƬ
		Mat image = cv::imread(jpgname);
		size_t h = image.rows;// ����(�߶�)
		size_t w = image.cols;// ��������ȣ�



		Mat inputBlob;
		inputBlob = cv::dnn::blobFromImage(image, 1.0, Size(600,400), Scalar(103.939, 116.779, 123.680), false, false);//
																		// ���м���
		net.setInput(inputBlob);
		Mat out = net.forward();

		Mat Styled;//4άת��3ά
		imagesFromBlob(out, Styled);
		// ���ͼƬ
		Styled /= 255;

		Mat uStyled;
		Styled.convertTo(uStyled, CV_8U, 255);//ת����ʽ
		cv::imshow("���ͼ��", uStyled);
		cv::imwrite("���ͼ��.jpg", uStyled);

		waitKey(0);
		return 0;
	}
