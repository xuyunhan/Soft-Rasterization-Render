#include <stdio.h>
#include "Math_Lib.h"
#include "Show_Window.h"
#include <Windows.h>
#include <time.h>
#include"FreeImage.h"

typedef struct
{
	int r, g, b;
}Color;

typedef struct
{
	float u,v;
}TexCoord;

typedef struct
{
	float r, g, b;
}Color_float;

typedef struct
{
	Pos pos;
	Color color;
	TexCoord texcoord;
	float rhw;
}Vertex;

typedef Pos Vector;//Pos当Vector用

Vertex cube[8] = 
{
	{ { 1, -1,  1, 1 },{ 1.0f, 0.2f, 0.2f } ,{0.0f,0.0f}},
	{ { -1, -1,  1, 1 },{ 0.2f, 1.0f, 0.2f } ,{ 0.0f,0.0f } },
	{ { -1,  1,  1, 1 },{ 0.2f, 0.2f, 1.0f } ,{ 0.0f,0.0f } },
	{ { 1,  1,  1, 1 },{ 1.0f, 0.2f, 1.0f } ,{ 0.0f,0.0f } },
	{ { 1, -1, -1, 1 },{ 1.0f, 1.0f, 0.2f } ,{ 0.0f,0.0f } },
	{ { -1, -1, -1, 1 },{ 0.2f, 1.0f, 1.0f } ,{ 0.0f,0.0f } },
	{ { -1,  1, -1, 1 },{ 1.0f, 0.3f, 0.3f } ,{ 0.0f,0.0f } },
	{ { 1,  1, -1, 1 },{ 0.2f, 1.0f, 0.3f } ,{ 0.0f,0.0f } }
};

Vertex cube_processed[8] =
{
	{ { 1, -1,  1, 1 },{ 1.0f, 0.2f, 0.2f } ,{ 0.0f,0.0f } },
	{ { -1, -1,  1, 1 },{ 0.2f, 1.0f, 0.2f } ,{ 0.0f,0.0f } },
	{ { -1,  1,  1, 1 },{ 0.2f, 0.2f, 1.0f } ,{ 0.0f,0.0f } },
	{ { 1,  1,  1, 1 },{ 1.0f, 0.2f, 1.0f } ,{ 0.0f,0.0f } },
	{ { 1, -1, -1, 1 },{ 1.0f, 1.0f, 0.2f } ,{ 0.0f,0.0f } },
	{ { -1, -1, -1, 1 },{ 0.2f, 1.0f, 1.0f } ,{ 0.0f,0.0f } },
	{ { -1,  1, -1, 1 },{ 1.0f, 0.3f, 0.3f } ,{ 0.0f,0.0f } },
	{ { 1,  1, -1, 1 },{ 0.2f, 1.0f, 0.3f } ,{ 0.0f,0.0f } }
};

// const int a = 800 * 600;
// Color_int frame_buffer[800][600];
// float z_buffer[800][600];

unsigned int **framebufferPtr;      // 像素缓存：framebuffePtr[y] 代表第 y行
float **zbufferPtr;//深度缓冲
float *zbuffer;
float zbufferMaxDepth[800][600];//用来memcpy重置zbuffer的
Color texbuffer[256][256];
float cube_camera_w[8];// 归一化前相机空间内顶点的w，暂存

typedef struct {
	matrix_t world;         // 世界坐标变换
	matrix_t view;          // 摄影机坐标变换
	matrix_t projection;    // 投影变换
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}	TransformMatrix;

TransformMatrix transformMatrix;

void ApplyHomogenize()//归一化到NDC
{
	float rhw = 0;
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_camera_w[i];
// 		cube_processed[i].rhw = rhw;
//		matrix_apply(& cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//到相机空间
		 cube_processed[i].pos.x = ( cube_processed[i].pos.x * rhw + 1.0f) * 800.0f * 0.5f;
		 cube_processed[i].pos.y = (1.0f -  cube_processed[i].pos.y * rhw) * 600.0f * 0.5f;
		 cube_processed[i].pos.z =  cube_processed[i].pos.z * rhw;
		 cube_processed[i].pos.w = cube_camera_w[i];
	};

}

void ApplyWVPTransform()//世界变换
{
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		matrix_apply(&cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//到CVV空间
	};
}

Color GetTexture(float u, float v)
{ 
// 	unsigned int nColorRef = RGB(0, 128, 255);//此时a为BGR
// 	int red = nColorRef & 255;//这样从COLORREF中获取RGB
// 	int green = nColorRef >> 8 & 255;
// 	int blue = nColorRef >> 16 & 255;
	if (u<0||v<0||u>255||v>255)
	{
		Color result = { 255, 0, 0 };
		return result;
	}
#if 1
// 	u *= 256.0f;
// 	v *= 256.0f;
	int x = floor(u);
	int y = floor(v);
	float u_ratio = u - x;
	float v_ratio = v - y;
	float u_opposite = 1 - u_ratio;
	float v_opposite = 1 - v_ratio;

	unsigned short r = ((texbuffer[x][y].r) * u_opposite + (texbuffer[x + 1][y].r) * u_ratio) * v_opposite + ((texbuffer[x][y + 1].r) * u_opposite + (texbuffer[x + 1][y + 1].r) * u_ratio) * v_ratio;
	unsigned short g = ((texbuffer[x][y] .g) * u_opposite + (texbuffer[x + 1][y] .g) * u_ratio) * v_opposite + ((texbuffer[x][y + 1] .g) * u_opposite + (texbuffer[x + 1][y + 1] .g) * u_ratio) * v_ratio;
	unsigned short b = ((texbuffer[x][y] .b) * u_opposite + (texbuffer[x + 1][y] .b) * u_ratio) * v_opposite + ((texbuffer[x][y + 1] .b) * u_opposite + (texbuffer[x + 1][y + 1] .b) * u_ratio) * v_ratio;
	Color result = { r,g,b };
	return result;

#endif // 0
 return texbuffer[(int)u][(int)v];
//	return a;

}


void Barycentric(Vertex* p1, Vertex* p2, Vertex* p3, float diffuse)
{
	int i = 0, j = 0 ,x= 0,y=0;
Vector dir_light = { -3,3,-10,1 };//向右边照的方向光
	float x1 = p1->pos.x;
	float y1 = p1->pos.y;
	float z1 = p1->pos.z;
	float x2 = p2->pos.x;
	float y2 = p2->pos.y;
	float z2 = p2->pos.z;
	float x3 = p3->pos.x;
	float y3 = p3->pos.y;
	float z3 = p3->pos.z;
	float a, b, c;//方程系数
//	平面可以由Ax + By + Cz + D = 0来表示，因此z = f(x, y) = (-Ax - By - D) / C。
//	可以观察到，f(x + 1, y) – f(x, y) = -A / C，f(x, y + 1) – f(x, y) = -B / C，因此不必针对每个点都计算z值，只要有了三角形一个顶点的z值就可以用加减法算出其它点的z值。
	float A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
	float B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
	float C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
	float D = -(A*x1 + B*y1 + C*z1);
	float deltaZperX = -A / C;//以（x1,y1,z1）为起始点
	float deltaZperY = -B / C;
	float z = 0;
	//纹理
	double u, v;
// 	float u1, v1, u2, v2, u3, v3;
	//光照
// 	Vector _2N = { normal->x * 2,normal->y * 2,normal->z * 2,0 };

// 	Vector p = { p1->pos.x / 1600.0f +p3->pos.x / 1600.0f, p1->pos.y / 1200.0f+ p3->pos.y / 1200.0f, (p1->pos.z +p3->pos.z)/2.0f ,0};
// 	Vector temp; matrix_t matVP,matWorldIdentity,t;
// 	matrix_set_identity(&matWorldIdentity);
// 	matrix_mul(&t, &transformMatrix.view, &matWorldIdentity);
// 	matrix_mul(&matVP, &t, &transformMatrix.projection);
// 	matrix_apply(&temp, &dir_light, &matVP);
// 
// 	vector_normalize(&temp);
// 	vector_normalize(&p);
// 	Vector L;  vector_sub(&L, &temp, &p);
// 	vector_normalize(&L);
// 	Vector R,temp;
// 	temp = vector_multply(normal, vector_dotproduct(&_2N, &L));
// 	vector_sub(&R, &temp, &L);
// 	double diffuse = max(vector_dotproduct(&L, normal), 0);
	//diffuse = min(diffuse, 1);
//	printf("%f\n", diffuse);
// 	Vector V = { 0 - p.x,0 - p.y,0 - p.z,0 - p.w };
// 	float VdotR = vector_dotproduct(&V, &R);
// 	float VdotR_n = VdotR;
// 	int n = 1;//次数
// 	VdotR_n = pow(VdotR, n);
// 
// 	float Ks = 1; float Is = 1;
// 	float Ispecular = Ks*Is*VdotR_n;
	//光照完
	//获得本三角形所在的矩形
	float xmin, xmax, ymin, ymax;
	xmin = x1 < x2 ? x1 : x2;
	xmin = xmin < x3 ? xmin : x3;
	xmax = x1 > x2 ? x1 : x2;
	xmax = xmax > x3 ? xmax : x3;
	ymin = y1 < y2 ? y1 : y2;
	ymin = ymin < y3 ? ymin : y3;
	ymax = y1 > y2 ? y1 : y2;
	ymax = ymax > y3 ? ymax : y3;

	int red = 0, green = 0, blue = 0;
	const int xend = xmax < 800 ? xmax : 800;
	const int yend = ymax < 600 ? ymax : 600;
	const int xstart = xmin > 0 ? xmin : 0;
	const int ystart = ymin > 0 ? ymin : 0;
	double zr = 0;
	Color texColor;
	const float p1w = p1->pos.w, p2w = p2->pos.w, p3w = p3->pos.w;
	const float p1u = p1->texcoord.u, p2u = p2->texcoord.u, p3u = p3->texcoord.u;
	const float p1v = p1->texcoord.v, p2v = p2->texcoord.v, p3v = p3->texcoord.v;
#pragma omp parallel for private(x,y) num_threads(4)
	for (x = xstart; x < xend; x++)
	{
		for (y = ystart; y < yend; y++)
		{
			c = ((y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			b = ((y1 - y3)*x + (x3 - x1)*y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			a = 1 - b - c;
			if (a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)
			{
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//算该三角形的Z值
				if (z < (zbuffer[(int)y * 800 + (int)x]))//若离屏幕更近则显示否则舍弃
				{
					zr = a*(1 / p1w) + b*(1 / p2w) + c*(1 / p3w);
					u = ((a*(p1u / p1w) + b*(p3u / p2w) + c*(p3u / p3w)) / zr) * 255.0;//w
					v = ((a*(p1v / p1w) + b*(p2v / p2w) + c*(p3v / p3w)) / zr) * 255.0;//h
					zbuffer[(int)y * 800 + (int)x] = z;

					const float inv255 = 1.0f / 255.0f;
					texColor = GetTexture(u, v);
					float r = (float)texColor.r*inv255*diffuse;
					float g = (float)texColor.g*inv255*diffuse;
					float b = (float)texColor.b*inv255*diffuse;

					framebufferPtr[y][x] = RGB(r*255.0f, g*255.0f, b*255.0f);
// 					framebufferPtr[x][y] = GetTexture(u, v);// *0x0000c0;
				}
			}
		}
	}
// 	getchar();
}

void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3, float diffuse)//绘制图元
{
// 	ApplyWorldTransform();//顶点坐标从局部坐标系转到世界坐标系
// 	ApplyViewTransform();//到相机坐标系
// 	ApplyProjectionTransform();//到CVV坐标系并透视除法（投影空间）
// 	ApplyHomogenize();//归一化到NDC

	//背面剔除
//	double z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x);
// 	已知三点求三角形法向量(a,b,c)
// 		a = ((p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y));
// 
// 		b = ((p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z));
// 
// 		c = ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));	
// 	Vector normal = { 0,0,0,1 };
// 	normal.x = (p2->pos.y - p1->pos.y) * (p3->pos.z - p1->pos.z) - (p3->pos.y - p1->pos.y) * (p2->pos.z - p1->pos.z);
// 	normal.y = (p2->pos.z - p1->pos.z) * (p3->pos.x - p1->pos.x) - (p3->pos.z - p1->pos.z) * (p2->pos.x - p1->pos.x);
// 	normal.z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p3->pos.x - p1->pos.x) * (p2->pos.y - p1->pos.y);
// 	vector_normalize(&normal);//光照模型里法线要规范化
// 	if ((p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p3->pos.x - p1->pos.x) * (p2->pos.y - p1->pos.y) > 0.0f)
	{
		//开始光栅化
		Barycentric(p1, p2, p3, diffuse);
	}
}

void DrawPlane(int a,int b,int c, int d)//绘制四边形，四个参数为顶点的索引
{

	Vertex* p1 = &cube_processed[a], *p2 = &cube_processed[b], *p3 = &cube_processed[c], *p4 =& cube_processed[d];
	p1->pos.w = cube_camera_w[a]; p2->pos.w = cube_camera_w[b]; p3->pos.w = cube_camera_w[c]; p4->pos.w = cube_camera_w[d];
// 	Vertex tp1, tp2, tp3,tp4;
// 	tp1.pos = p1->pos; tp2.pos = p2->pos; tp3.pos = p3->pos; tp4.pos = p4->pos;
// 	tp1.rhw = p1->rhw; tp2.rhw = p2->rhw; tp3.rhw = p3->rhw; tp4.rhw = p4->rhw;
// 	tp1.pos.w = tp2.pos.w = cube_camera_w[b]; tp3.pos.w = cube_camera_w[c]; tp4.pos.w = cube_camera_w[d];
	p1->texcoord.u = 0, p1->texcoord.v = 0, p2->texcoord.u = 0, p2->texcoord.v = 1;
	p3->texcoord.u = 1, p3->texcoord.v = 1, p4->texcoord.u = 1, p4->texcoord.v = 0;
// 	tp1.texcoord = p1->texcoord; tp2.texcoord = p2->texcoord; tp3.texcoord = p3->texcoord; tp4.texcoord = p4->texcoord;
	//在这里就把光照算好算了...一个平面一个光照，对平行光来说
	Pos normal = { 0,0,0,0 };
	Pos* op1 = &cube[a].pos, *op2 = &cube[b].pos, *op3 = &cube[c].pos, *op4 = &cube[d].pos;

	// 	已知三点求三角形法向量(a,b,c)
// 		a = ((p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y));
// 		b = ((p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z));
// 		c = ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));	
	// 	//这个算出的op面法向量是对的，按上面的公式法向量反了
	normal.x = (op2->y - op3->y) * (op1->z - op3->z) - (op1->y - op3->y) * (op2->z - op3->z);
	normal.y = (op2->z - op3->z) * (op1->x - op3->x) - (op1->z - op3->z) * (op2->x - op3->x);
	normal.z = (op2->x - op3->x) * (op1->y - op3->y) - (op1->x - op3->x) * (op2->y - op3->y);
	Vector L; Vector normal_rotated;
	Vector dir_light = { 3,0,-0,0 };//向屏幕内照的方向光

	matrix_apply(&normal_rotated, &normal, &transformMatrix.world);
	Vector normal_cameraspace;
	matrix_apply(&normal_cameraspace, &normal, &transformMatrix.transform);
	if (normal_cameraspace.z > 0)
	{
		return;
	}
	vector_normalize(&normal_rotated);//光照模型里法线要规范化
	Vector p = { (op1->x + op2->x + op3->x + op4->x)*0.25f, (op1->y + op2->y + op3->y + op4->y)*0.25f, (op1->z + op2->z + op3->z + op4->z)*0.25f, 1 };
	Vector p_rotated;
	matrix_apply(&p_rotated, &p, &transformMatrix.world);
	vector_sub(&L, &dir_light, &p_rotated);
	vector_normalize(&L);
// 	Phong模型认为镜面反射的光强与反射光线和视线的夹角相关：
// 		Ispec = Ks * Il * (dot(V, R)) ^ Ns
// 		其中Ks 为镜面反射系数,Il是光源强度，Ns是高光指数，V表示从顶点到视点的观察方向，R代表反射光方向。由于反射光的方向R可以通过入射光方向L(从顶点指向光源)和物体的法向量求出，
// 		R + L = 2 * dot(N, L) * N  即 R = 2 * dot(N, L) * N - L
// 		所以最终的计算式为：
// 		Ispec = Ks * Il * (dot(V, (2 * dot(N, L) * N – L)) ^ Ns
		float Ks = 0.7f, Il = 1.0f,Ns = 20.0f;
		Vector V; vector_sub(&V, &dir_light, &p_rotated);
		vector_normalize(&V);
		Vector t = vector_multply(&normal_rotated, 2.0f*vector_dotproduct(&L, &normal_rotated));
		Vector R; vector_sub(&R, &t, &L);
		vector_normalize(&R);
		float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
		Ispec = max(Ispec, 0);
// 		Ispec = min(Ispec, 1);
		// 	Vector R,temp;
// 	temp = vector_multply(normal, vector_dotproduct(&_2N, &L));
// 	vector_sub(&R, &temp, &L);
	float diffuse = vector_dotproduct(&L, &normal_rotated);
	diffuse = max(diffuse, 0);
// 	diffuse = min(diffuse, 1);

	DrawPrimitive(p3, p2, p1, diffuse);//分成三角形图元绘制，逆时针为绘制方式
	DrawPrimitive(p1, p4, p3, diffuse);
}

void UpdateMVPMatrix()
{
	matrix_t m;
	matrix_mul(&m, &transformMatrix.world, &transformMatrix.view);
	matrix_mul(&transformMatrix.transform, &m, &transformMatrix.projection);// transform = world * view * projection
}
static int _switch = 1;
int FPS = 0;

void DrawCube()
{
	int x, y;
	ApplyWVPTransform();//顶点坐标从局部坐标系转到裁剪后的NDC
	for (size_t i = 0; i < 8; i++)
	{
		cube_camera_w[i] = cube_processed[i].pos.w;
	}
	//乘透视变换矩阵后到了CVV所在的空间，由于CVV是个立方体，可以很方便的裁剪。但是CVV是个立方体，所以长宽比这时候会变化，需要后续修正
	ApplyHomogenize();//归一化到NDC
	//开始光栅化

	DrawPlane(2, 3, 0, 1);
	DrawPlane(0, 4, 5, 1);
	DrawPlane(1, 5, 6, 2);
	DrawPlane(3, 2, 6, 7);
	DrawPlane(7, 4, 0, 3);//一开始面向屏幕
	DrawPlane(7, 6, 5, 4);//

	FPS++;
}


void RotateCube(float angle)
{
	matrix_t m;
// 	matrix_set_rotate(&m, 0, 1, 0, angle);//沿y轴转
// 	matrix_set_rotate(&m, 1,0 , 0, angle);//沿x轴转
// 	 matrix_set_rotate(&m,  0, 0,1, angle);//沿z轴转
	matrix_set_rotate(&m, -1, -0.5, 1, angle);//算旋转矩阵
	transformMatrix.world = m;
	UpdateMVPMatrix();//更新矩阵
}

void SetCameraLookAt(TransformMatrix *transformMatrix, float x, float y, float z) {//todo::这个函数有个问题，lookat算出来的view矩阵不太对，可以固定相机位置在（0,3,0），只绘制（3,2,7,6）这个面的某个三角形来重现，在MVP变换后，归一化之前，CVV里的坐标就错了
	point_t eye = { x, y, z, 1 }, at = { 0, 0, 0, 1 }, up = { 0, 0, 1, 0 };//y向上
	matrix_set_identity(&transformMatrix->world);
	matrix_set_identity(&transformMatrix->view);
	matrix_set_perspective(&transformMatrix->projection, 3.1415926f * 0.5f, 800.0f/600.0f, 1.0f, 500.0f);
	matrix_set_lookat(&transformMatrix->view, &eye, &at, &up);
	UpdateMVPMatrix();
}

void BindFB(int width, int height, void *fb) {
// 	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
// 	char *ptr = (char*)malloc(need + 64);
	char *framebuf, *zbuf;
	int i,j;
	for (size_t i = 0; i < 800; i++)
	{
		for (size_t j = 0; j < 600; j++)
		{
			zbufferMaxDepth[i][j] = 1.0f;
		}
	}
// 	assert(ptr);
// 	framebufferPtr = (unsigned int **)ptr;//原来的
	framebufferPtr = (unsigned int **)malloc(sizeof(unsigned int*) * height);

// 	zbuffer = (float**)(ptr + sizeof(void*) * height);
	zbufferPtr = (float**)malloc(sizeof(float*) * height);
	zbuffer = (float*)malloc(800 * 600 * sizeof(float));
//	memset(zbuffer, 1, 800 * 600 * sizeof(float));
	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
// 	ptr += sizeof(void*) * height * 2;
// 	device->texture = (IUINT32**)ptr;
// 	ptr += sizeof(void*) * 1024;
// 	framebuf = (char*)ptr;
 //	zbuf = (char*)ptr + width * height * 4;
	zbuf = (char*)zbufferPtr;
// 	ptr += width * height * 8;
	//xuyunhan
//  	screen_fb = (char*)malloc(800 * 600 * 4);
// 	BITMAPINFO bi = { { sizeof(BITMAPINFOHEADER), 800, -600, 1, 32, BI_RGB,	800 * 600 * 4, 0, 0, 0, 0 } };
// 	screen_hb = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &screen_fb, 0, 0);

// 	memset(screen_fb, 1, 800 * 600 * 4);

// 	if (fb != NULL)
		framebuf = screen_fb;//framebuf指向fb的首地址

	for (j = 0; j < 600; j++)
	{
		framebufferPtr[j] = (unsigned int *)(framebuf + 800 * 4 * j);//framebuffer的每一行指向fb的每一行对应的地址
		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
	}

	//贴图部分
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;

		Color white = { 255,255,255 }, blue = {239,190,63 }, blue2 = { 239,0,0 };
//		texbuffer[j][i] = ((x + y) & 1) ? 0xffffff : RGB(255,0,0);
		texbuffer[j][i] = ((x + y) & 1) ? white : blue;
		}
	}

/*	device->texture[0] = (IUINT32*)ptr;
	device->texture[1] = (IUINT32*)(ptr + 16);
	memset(device->texture[0], 0, 64);
	device->tex_width = 2;
	device->tex_height = 2;
	device->max_u = 1.0f;
	device->max_v = 1.0f;
	device->width = width;
	device->height = height;
	device->background = 0xc0c0c0;
	device->foreground = 0;
	transform_init(&device->transform, width, height);
	*/
}

void ClearFrameBuffer()
{
	memset(screen_fb, 0xffffff, 800 * 600 * 4);
	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
}

float cam_angle = 0, self_angle = 0;

void CALLBACK timerProc(HWND a, UINT b, UINT c, DWORD d)
{
	printf("FPS: %d\n", FPS);
	FPS = 0;
// 	self_angle += 0.1f;
// 	ClearFrameBuffer();
// 	RotateCube(self_angle);
}

int  main()
{
	screen_init(800, 600, _T("Cube"));
	BindFB(800, 600, screen_fb);

	static float x = 3, y = 0, z = 0;
	SetTimer(NULL,1, 1000, timerProc);
	SetCameraLookAt(&transformMatrix, 3, 0, 0);
	while (1/*screen_exit == 0 && screen_keys[VK_ESCAPE] == 0*/)
	{
		if (screen_keys[VK_SPACE])
			_switch = -_switch;
// 		if (_switch == 1)
// 			SetCameraLookAt(&transformMatrix, 3, 0, 0);
// 		else if (_switch == -1)
// 			SetCameraLookAt(&transformMatrix, -3, 0, 0);
		if (screen_keys[VK_UP])
		{
			x -= 0.2;
			SetCameraLookAt(&transformMatrix, x, 0, 0);
		}
		if (screen_keys[VK_DOWN])
		{
			x += 0.2;
			SetCameraLookAt(&transformMatrix, x, 0, 0);
		}
	{
			self_angle += 0.005f;
// 			self_angle -=90.0f;
			RotateCube(self_angle);
//		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
		}
// 		if (screen_keys[VK_RIGHT])
// 		{
// 			self_angle -= 0.1f;
// 			RotateCube(self_angle);
// //		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
// 			ClearFrameBuffer();
// 		}

		DrawCube();

		screen_update();
// 		getchar();
		ClearFrameBuffer();
// 		Sleep(100);
	}
	return 0;
}