#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/** @function main */
int main()
{

	Mat src;
	Mat grad;
	char* window_name = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

	/// װ��ͼ��
	src = imread("../FFT_20.png");

	if (!src.data)
	{
		return -1;
	}

	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// ������ʾ����
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// ���� grad_x �� grad_y ����
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// �� X�����ݶ�
	//Scharr( src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// ��Y�����ݶ�
	//Scharr( src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// �ϲ��ݶ�(����)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	imshow(window_name, grad);
	imwrite("C:/Users/Administrator/Desktop/Gradient_20.png", grad);
	imwrite("C:/Users/Administrator/Desktop/Gradient_20_x.png", abs_grad_x);
	imwrite("C:/Users/Administrator/Desktop/Gradient_20_y.png", abs_grad_y);
	waitKey(0);

	return 0;
}