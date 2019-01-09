//
#include "MSRCR.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <io.h>
#include <Windows.h>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <direct.h>

using namespace cv;
using namespace std;
#define TWO_PI 6.2831853071795864769252866
typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;
typedef struct matrix{
	int rows, cols;
	float **vals;
} matrix;
typedef struct{
	float x, y, w, h;
} box;
typedef struct{
	int w, h;
	matrix X;
	matrix y;
	int shallow;
	int *num_boxes;
	box **boxes;
} data;
typedef struct load_args{
	int threads;
	char **paths;
	char *path;
	int n;
	int m;
	char **labels;
	int h;
	int w;
	int c; // color depth
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min, max, size;
	int classes;
	int background;
	int scale;
	int small_object;
	float jitter;
	int flip;
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	data *d;
	image *im;
	image *resized;
	//data_type type;
	//tree *hierarchy;
} load_args;


extern int readDir(char *dirName, vector<string> &filesName);
unsigned int random_gen1()
{
	unsigned int rnd = 0;

	rnd = rand();

	return rnd;
}
float random_float1()
{

	return ((float)random_gen1() / (float)RAND_MAX);

}

float rand_uniform_strong1(float min, float max)
{
	if (max < min) 
	{
		float swap = min;
		min = max;
		max = swap;
	}
	return (random_float1() * (max - min)) + min;
}
float rand_scale1(float s)
{
	float scale = rand_uniform_strong1(1, s);
	if (random_gen1() % 2) return scale;
	return 1. / scale;
}
int rand_int1(int min, int max)
{
	if (max < min)
	{
		int s = min;
		min = max;
		max = s;
	}
	int r = (rand() % (max - min + 1)) + min;
	return r;
}
float rand_uniform1(float min, float max)
{
	if (max < min)
	{
		float swap = min;
		min = max;
		max = swap;
	}
	return ((float)rand() / RAND_MAX * (max - min)) + min;
	//return (random_float() * (max - min)) + min;
}
image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}
image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}
static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
	if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
	/*
	if(x < 0) x = 0;
	if(x >= m.w) x = m.w-1;
	if(y < 0) y = 0;
	if(y >= m.h) y = m.h-1;
	*/
	if (c < 0 || c >= m.c) return 0;
	return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
		return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] += val;
}
float bilinear_interpolate(image im, float x, float y, int c)
{
	int ix = (int)floorf(x);
	int iy = (int)floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
		dy     * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
		(1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
		dy     *   dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
	return val;
}
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
	int x, y, c;
	float cx = im.w / 2.;
	float cy = im.h / 2.;
	image rot = make_image(w, h, im.c);
	for (c = 0; c < im.c; ++c)
	{
		for (y = 0; y < h; ++y)
		{
			for (x = 0; x < w; ++x)
			{
				float rx = cos(rad)*((x - w / 2.) / s*aspect + dx / s*aspect) - sin(rad)*((y - h / 2.) / s + dy / s) + cx;
				float ry = sin(rad)*((x - w / 2.) / s*aspect + dx / s*aspect) + cos(rad)*((y - h / 2.) / s + dy / s) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}
image random_augment_image(image im, Mat src, Mat & dst,  float rad, float aspect, int r, int size)
{
	
	int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
	float scale = (float)r / min;

	

	float dx = (im.w*scale / aspect - size) / 2.;
	float dy = (im.h*scale - size) / 2.;

	/*if (dx < 0) 
		dx = 0;
	if (dy < 0) 
		dy = 0;*/

	dx = rand_uniform1(-dx, dx);
	dy = rand_uniform1(-dy, dy);

	printf("dx=%0.2f,dy=%0.2f\n",dx,dy);

	image crop = rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);

	return crop;
}
float three_way_max(float a, float b, float c)
{
	return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
	return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}
void rgb_to_hsv(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
			r = get_pixel(im, i, j, 0);
			g = get_pixel(im, i, j, 1);
			b = get_pixel(im, i, j, 2);
			float max = three_way_max(r, g, b);
			float min = three_way_min(r, g, b);
			float delta = max - min;
			v = max;
			if (max == 0){
				s = 0;
				h = 0;
			}
			else{
				s = delta / max;
				if (r == max){
					h = (g - b) / delta;
				}
				else if (g == max) {
					h = 2 + (b - r) / delta;
				}
				else {
					h = 4 + (r - g) / delta;
				}
				if (h < 0) h += 6;
				h = h / 6.;
			}
			set_pixel(im, i, j, 0, h);
			set_pixel(im, i, j, 1, s);
			set_pixel(im, i, j, 2, v);
		}
	}
}

void hsv_to_rgb(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	float f, p, q, t;
	for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
			h = 6 * get_pixel(im, i, j, 0);
			s = get_pixel(im, i, j, 1);
			v = get_pixel(im, i, j, 2);
			if (s == 0) {
				r = g = b = v;
			}
			else {
				int index = floor(h);
				f = h - index;
				p = v*(1 - s);
				q = v*(1 - s*f);
				t = v*(1 - s*(1 - f));
				if (index == 0){
					r = v; g = t; b = p;
				}
				else if (index == 1){
					r = q; g = v; b = p;
				}
				else if (index == 2){
					r = p; g = v; b = t;
				}
				else if (index == 3){
					r = p; g = q; b = v;
				}
				else if (index == 4){
					r = t; g = p; b = v;
				}
				else {
					r = v; g = p; b = q;
				}
			}
			set_pixel(im, i, j, 0, r);
			set_pixel(im, i, j, 1, g);
			set_pixel(im, i, j, 2, b);
		}
	}
}
void scale_image_channel(image im, int c, float v)
{
	int i, j;
	for (j = 0; j < im.h; ++j)
	{
		for (i = 0; i < im.w; ++i)
		{
			float pix = get_pixel(im, i, j, c);
			pix = pix*v;
			set_pixel(im, i, j, c, pix);
		}
	}
}
void constrain_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h*im.c; ++i)
	{
		if (im.data[i] < 0)
			im.data[i] = 0;
		if (im.data[i] > 1) 
			im.data[i] = 1;
	}
}
void distort_image(image im, float hue, float sat, float val)
{
	if (im.c >= 3)
	{
		rgb_to_hsv(im);
		scale_image_channel(im, 1, sat);
		scale_image_channel(im, 2, val);
		int i;
		for (i = 0; i < im.w*im.h; ++i)
		{
			im.data[i] = im.data[i] + hue;
			if (im.data[i] > 1) im.data[i] -= 1;
			if (im.data[i] < 0) im.data[i] += 1;
		}
		hsv_to_rgb(im);
	}
	else
	{
		scale_image_channel(im, 0, val);
	}
	constrain_image(im);
}
void random_distort_image(image im, float hue, float sat, float exp)
{
	
	distort_image(im, hue, sat, exp);
}
void free_image(image m)
{
	if (m.data)
	{
		free(m.data);
	}
}
image load_image_augment_paths(image im,Mat src, Mat & dst, int r, int size, float rad, float aspect, float hue, float saturation, float exposure)
{
	
		image crop = random_augment_image(im,src, dst,rad, aspect, r, size);
		
		random_distort_image(crop, hue, saturation, exposure);

		
	return crop;
}
image load_data_augment(image im, Mat src, Mat & dst, int r, int size, float rad, float aspect, float hue, float saturation, float exposure)
{
	//data d;
	//d.X = 
	image img=load_image_augment_paths(im, src, dst,  r, size, rad, aspect, hue, saturation, exposure);
	
	return img;
}
image resize_image(image im, int w, int h)
{
	image resized = make_image(w, h, im.c);
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k){
		for (r = 0; r < im.h; ++r){
			for (c = 0; c < w; ++c){
				float val = 0;
				if (c == w - 1 || im.w == 1){
					val = get_pixel(im, im.w - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k){
		for (r = 0; r < h; ++r){
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c){
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1) continue;
			for (c = 0; c < w; ++c){
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}
image ipl_to_image(IplImage* src)
{
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;
	image out = make_image(w, h, c);
	int i, j, k, count = 0;;

	for (k = 0; k < c; ++k)
	{
		for (i = 0; i < h; ++i)
		{
			for (j = 0; j < w; ++j)
			{
				out.data[count++] = data[i*step + j*c + k] / 255.;
			}
		}
	}
	return out;
}
void rgbgr_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i)
	{
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}
image load_image_cv(char *filename, int channels)
{
	IplImage* src = 0;
	int flag = -1;
	if (channels == 0) flag = 1;
	else if (channels == 1) flag = 0;
	else if (channels == 3) flag = 1;
	else 
	{
		fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
	}

	if ((src = cvLoadImage(filename, flag)) == 0)
	{
		char shrinked_filename[1024];
		if (strlen(filename) >= 1024) sprintf(shrinked_filename, "name is too long");
		else sprintf(shrinked_filename, "%s", filename);
		fprintf(stderr, "Cannot load image \"%s\"\n", shrinked_filename);
		FILE* fw = fopen("bad.list", "a");
		fwrite(shrinked_filename, sizeof(char), strlen(shrinked_filename), fw);
		char *new_line = "\n";
		fwrite(new_line, sizeof(char), strlen(new_line), fw);
		fclose(fw);

		/*if (check_mistakes) 
			getchar();*/

		return make_image(10, 10, 3);
		//exit(EXIT_FAILURE);
	}
	image out = ipl_to_image(src);
	cvReleaseImage(&src);
	if (out.c > 1)
		rgbgr_image(out);
	return out;
}
image load_image(char *filename, int w, int h, int c)
{


	image out = load_image_cv(filename, c);        // OpenCV 2.4.x

	return out;
}
image load_image_color(char *filename, int w, int h)
{
	return load_image(filename, w, h, 3);
}
image copy_image(image p)
{
	image copy = p;
	copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
	return copy;
}
void save_image_jpg(image p, const char *name)
{
	image copy = copy_image(p);
	if (p.c == 3) rgbgr_image(copy);
	int x, y, k;

	//char buff[256];
	//sprintf(buff, "%s.jpg", name);

	IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y)
	{
		for (x = 0; x < p.w; ++x)
		{
			for (k = 0; k < p.c; ++k)
			{
				disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
			}
		}
	}
	cvSaveImage(name, disp, 0);
	cvReleaseImage(&disp);
	free_image(copy);
}
int test54_1(int argc, char *argv[])
{



	srand((unsigned)time(NULL));

	string imgpath1 = "E:\\wusuizhedangqufen\\pachong\\biaoji\\patupian1";
	string savepath = "E:\\wusuizhedangqufen\\pachong\\biaojidata\\patupian1";
	mkdir(savepath.c_str());
	string drawpath = "E:\\wusuizhedangqufen\\pachong\\biaojidata\\patupian1draw";
	mkdir(drawpath.c_str());
	int nrotate = 10;//旋转个数
	int nmove = 10;//平移个数
	int ncrop = 10;//裁剪个数
	int nmsrcr = 10;//msrcr个数
	int ncolor = 10;//色彩抖动
	vector<string> v1_img_;
	readDir((char*)imgpath1.c_str(), v1_img_);

	for (int i = 0; i < v1_img_.size(); i++)
	{
		//string imagename = v1_img_[i];

		string imagename = "E:\\wusuizhedangqufen\\pachong\\biaoji\\patupian1\\2.jpg";

		int npos = imagename.find_last_of('\\');
		int npos2 = imagename.find_last_of('.');

		string name1 = imagename.substr(npos + 1, npos2 - npos - 1);
		Mat img = imread(imagename.c_str());

		cout << imagename.c_str() << endl;

		if (img.data == NULL)
		{
			printf("img.data = NULL！\n");
			system("pause");
			continue;
		}

		float jitter = 0.3;
		int use_flip = 1;
		float hue = 0.1;
		float saturation = 1.5;
		float	exposure = 1.5;
		float  aspect = 1.5;
		int angle = 10;
		int min = 300;
		int max = 500;
		int size = 600;


		int oh = img.rows;
		int ow = img.cols;

		int dw = (ow*jitter);
		int dh = (oh*jitter);
		image im = load_image_color((char*)imagename.c_str(), 0, 0);
		for (int j = 0; j < 10; j++)
		{
			
			printf("-------%d-------------\n", j);

			Mat src = img.clone();
			Mat dst = src.clone();
			//load_args a = {0};
			image iimg = copy_image(im);
			
			
			float asp = rand() / double(RAND_MAX);
			float asp1 = asp*(aspect - 1) + 1;
			if (rand() % 2 == 1)
			{
				asp1 = asp1;
			}
			else
			{
				asp1 = 1 * 1.0 / asp1;
			}

			float r1 = rand()%(max-min)+min;

			float ang = rand() % (2*angle)-angle ;
			float rad = ang * TWO_PI / 360.;

			float fhue = rand() / double(RAND_MAX);
			float dhue1 = fhue * (0.11) - 0.03;//[-0.03,0.08]

			float fsat = rand() / double(RAND_MAX);
			float dsat1 = fsat*(saturation - 1) + 1;
			if (rand() % 2 == 1)
			{
				dsat1 = dsat1;
			}
			else
			{
				dsat1 = 1 * 1.0 / dsat1;
			}
	
			float fexp = rand() / double(RAND_MAX);
			float dexp1 = fexp*(exposure - 1) + 1;
			if (rand() % 2 == 1)
			{
				dexp1 = dexp1;
			}
			else
			{
				dexp1 = 1 * 1.0 / dexp1;
			}



			printf("asp1=%0.2f,r1=%0.2f,rad=%0.2f\n", asp1, r1, rad);
			printf("dhue1=%0.2f,dsat1=%0.2f,dexp1=%0.2f\n", dhue1, dsat1, dexp1);

			image idst=load_data_augment(iimg, img, dst, r1, size, rad, asp1, dhue1, dsat1, dexp1);
			
			
			
			char file[1024];
			sprintf(file, "%s\\%s-%d.jpg", savepath.c_str(), name1.c_str(), j);
			//imwrite(file, dst);
			save_image_jpg(idst, file);


			free_image(iimg);
			free_image(idst);

			int jj = 0;
		}

		free_image(im);





		int jjjj = 0;
	}
	return 0;
}









