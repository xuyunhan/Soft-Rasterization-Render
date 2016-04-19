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
	unsigned int r, g, b;
}Color_int;

typedef struct
{
	Pos pos;
	Color color;
	TexCoord texcoord;
	float rhw;
}Vertex;

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
unsigned int texbuffer[256][256];
float cube_camera_w[8];// 归一化前相机空间内顶点的w，暂存

typedef struct {
	matrix_t world;         // 世界坐标变换
	matrix_t view;          // 摄影机坐标变换
	matrix_t projection;    // 投影变换
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}	TransformMatrix;

TransformMatrix transformMatrix;

void ApplyClipping()//裁剪变换
{
	;
}
void ApplyHomogenize()//归一化到NDC
{
	float rhw = 0;
	for (size_t i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_camera_w[i];
// 		cube_processed[i].rhw = rhw;
//		matrix_apply(& cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//到相机空间
		 cube_processed[i].pos.x = ( cube_processed[i].pos.x * rhw + 1.0f) * 800.0f * 0.5f;
		 cube_processed[i].pos.y = (1.0f -  cube_processed[i].pos.y * rhw) * 600.0f * 0.5f;
		 cube_processed[i].pos.z =  cube_processed[i].pos.z * rhw;
		 cube_processed[i].pos.w = 1.0f;
	};

}

void ApplyWVPTransform()//世界变换
{
	for (size_t i = 0; i < 8; i++)
	{
		matrix_apply(&cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//到CVV空间
	};
	ApplyClipping();//裁剪
}

unsigned int GetTexture(float u, float v)
{
	return texbuffer[(int)u][(int)v];
}

void vertex_rhw_init(Vertex *v) {
	float rhw = 1.0f / v->pos.w;
	v->rhw = rhw;
	v->texcoord.u *= rhw;
	v->texcoord.v *= rhw;
// 	v->color.r *= rhw;
// 	v->color.g *= rhw;
// 	v->color.b *= rhw;
}

void Barycentric(Vertex* p1, Vertex* p2, Vertex* p3, int color)
{
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
	float u, v;
// 	float u1, v1, u2, v2, u3, v3;

	for (int x = 0; x < 550; x++)
	{
		for (int y = 0; y <600; y++)
		{
			c = ((y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			b = ((y1 - y3)*x + (x3 - x1)*y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			a = 1 - b - c;
			if (a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)
			{
//				Color c = a*c1 + b*c2 + c*c3;
//				DrawPixel(x, y, c);
//				printf("点：x: %d y: %d\n", x, y);
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//算该三角形的Z值
				if (z < (zbuffer[(int)x * 800 + (int)y]))//若离屏幕更近则显示否则舍弃
				{
					double zr = a*(1 / p1->pos.w) + b*(1 / p2->pos.w) + c*(1 / p3->pos.w);
					u = ((a*(p1->texcoord.u / p1->pos.w) + b*(p3->texcoord.u / p2->pos.w) + c*(p3->texcoord.u / p3->pos.w)) / zr) * 255.0f;//w
					v = ((a*(p1->texcoord.v / p1->pos.w) + b*(p2->texcoord.v / p2->pos.w) + c*(p3->texcoord.v / p3->pos.w)) / zr) * 255.0f;//h
					zbuffer[(int)x*800+(int)y] = z;
				//	framebufferPtr[x][y] = color;// SetPixel(GetCDC(), x, y, RGB(255, 255, 255));
					framebufferPtr[x][y] = GetTexture(u, v);
				}
			}
		}
	}

// 	getchar();
}
void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3,int color)//绘制图元
{
// 	ApplyWorldTransform();//顶点坐标从局部坐标系转到世界坐标系
// 	ApplyViewTransform();//到相机坐标系
// 	ApplyProjectionTransform();//到CVV坐标系并透视除法（投影空间）
// 	ApplyHomogenize();//归一化到NDC

	//背面剔除
//	double z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x);
	if ((p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x) > 0.0f)
	{

		//开始光栅化
		Barycentric(p1, p2, p3, color);
	}
}

void DrawPlane(int a,int b,int c, int d,int color)//绘制四边形，四个参数为顶点的索引
{
	Vertex* p1 = &cube_processed[a], *p2 = &cube_processed[b], *p3 = &cube_processed[c], *p4 =& cube_processed[d];
	Vertex tp1, tp2, tp3,tp4;
	tp1.pos = p1->pos; tp2.pos = p2->pos; tp3.pos = p3->pos; tp4.pos = p4->pos;
	tp1.rhw = p1->rhw; tp2.rhw = p2->rhw; tp3.rhw = p3->rhw; tp4.rhw = p4->rhw;
	tp1.pos.w = cube_camera_w[a]; tp2.pos.w = cube_camera_w[b]; tp3.pos.w = cube_camera_w[c]; tp4.pos.w = cube_camera_w[d];
	p1->texcoord.u = 0, p1->texcoord.v = 0, p2->texcoord.u = 0, p2->texcoord.v = 1;
	p3->texcoord.u = 1, p3->texcoord.v = 1, p4->texcoord.u = 1, p4->texcoord.v = 0;
	tp1.texcoord = p1->texcoord; tp2.texcoord = p2->texcoord; tp3.texcoord = p3->texcoord; tp4.texcoord = p4->texcoord;

	DrawPrimitive(&tp3, &tp2, &tp1, color);//分成三角形图元绘制
	DrawPrimitive(&tp1, &tp4, &tp3, color);
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[a].pos.x, cube_processed[a].pos.y, cube_processed[a].pos.z, cube_processed[a].pos.w);
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[b].pos.x, cube_processed[b].pos.y, cube_processed[b].pos.z, cube_processed[b].pos.w);
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[c].pos.x, cube_processed[c].pos.y, cube_processed[c].pos.z, cube_processed[c].pos.w);
// 	printf("\n");
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[c].pos.x, cube_processed[c].pos.y, cube_processed[c].pos.z, cube_processed[c].pos.w);
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[d].pos.x, cube_processed[d].pos.y, cube_processed[d].pos.z, cube_processed[d].pos.w);
// 	printf("三角形屏幕坐标：x: %f y: %f z: %f w: %f\n", cube_processed[a].pos.x, cube_processed[a].pos.y, cube_processed[a].pos.z, cube_processed[a].pos.w);
// 	printf("\n");
}

void UpdateMVPMatrix()
{
	matrix_t m;
	matrix_mul(&m, &transformMatrix.world, &transformMatrix.view);
	matrix_mul(&transformMatrix.transform, &m, &transformMatrix.projection);// transform = world * view * projection
}

void DrawCube(float self_angle, float cam_angle)
{
	ApplyWVPTransform();//顶点坐标从局部坐标系转到裁剪后的NDC
	for (size_t i = 0; i < 8; i++)
	{
		cube_camera_w[i] = cube_processed[i].pos.w;
	};
	//乘透视变换矩阵后到了CVV所在的空间，由于CVV是个立方体，可以很方便的裁剪。但是CVV是个立方体，所以长宽比这时候会变化，需要后续修正
	ApplyHomogenize();//归一化到NDC
	//开始光栅化
	DrawPlane(2, 3, 0, 1, 100);//原来的
	DrawPlane(7, 6, 5, 4, -250);
	DrawPlane(0, 4, 5, 1, -50);
	DrawPlane(1, 5, 6, 2, 200);
	DrawPlane(3, 2, 6, 7, -550);
	DrawPlane(3, 7, 4, 0, 0);

// 	DrawPlane(2, 1, 0, 3, 100);


// 	DrawPlane( 0, 1, 2, 3, 0);
// 	DrawPlane( 4, 5, 6, 7, 0);
// 	DrawPlane( 0, 4, 5, 1, 0);
// 	DrawPlane( 1, 5, 6, 2, 0);
// 	DrawPlane( 2, 6, 7, 3, 0);
// 	DrawPlane( 3, 7, 4, 0, 0);

}

void RotateCube(float angle)
{
	matrix_t m;
	matrix_set_rotate(&m, -1, -0.5,1, angle);//算旋转矩阵
	transformMatrix.world = m;
	UpdateMVPMatrix();//更新矩阵
}

void SetCameraLookAt(TransformMatrix *transformMatrix, float x, float y, float z) {
	point_t eye = { x, y, z, 1 }, at = { 0, 0, 0, 1 }, up = { 0, 0, 1, 1 };
	matrix_set_identity(&transformMatrix->world);
	matrix_set_identity(&transformMatrix->view);
	matrix_set_perspective(&transformMatrix->projection, 3.1415926f * 0.5f, 800.0f/600.0f, 1.0f, 500.0f);
	matrix_set_lookat(&transformMatrix->view, &eye, &at, &up);
	UpdateMVPMatrix();
}

void BindFB(int width, int height, unsigned int *fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
	char *ptr = (char*)malloc(need + 64);
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
	framebufferPtr = (unsigned int **)ptr;//原来的
//	framebuffer = (unsigned int **)malloc(sizeof(unsigned int*) * height);
	for (j = 0; j < height; j++)
	{
//		framebuffer[j] = (unsigned int *)malloc(sizeof(void*));//framebuffer的每一行指向fb的每一行对应的地址
	}

// 	zbuffer = (float**)(ptr + sizeof(void*) * height);
	zbufferPtr = (float**)malloc(sizeof(float*) * height);
	zbuffer = (float*)malloc(800 * 600 * sizeof(float));
//	memset(zbuffer, 1, 800 * 600 * sizeof(float));
	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
//	ptr += sizeof(void*) * height * 2;
// 	device->texture = (IUINT32**)ptr;
//	ptr += sizeof(void*) * 1024;
//	framebuf = (char*)ptr;
 //	zbuf = (char*)ptr + width * height * 4;
	zbuf = (char*)zbufferPtr;
// 	ptr += width * height * 8;
// 	if (fb != NULL)
		framebuf = (unsigned int*)fb;//framebuf指向fb的首地址
	for (j = 0; j < height; j++)
	{
		framebufferPtr[j] = (unsigned int *)(framebuf + width * 4 * j);//framebuffer的每一行指向fb的每一行对应的地址
 		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
	}

	//贴图部分
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;
			texbuffer[j][i] = ((x + y) & 1) ? 0xffffff : 0x3fbcef;
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
	memset(screen_fb, 0xc0c0c0, 800 * 600 * 4);
	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
}

float cam_angle = 0, self_angle = 0;

void CALLBACK timerProc(HWND a, UINT b, UINT c, DWORD d)
{
	self_angle += 0.1f;
	ClearFrameBuffer();
	RotateCube(self_angle);
}
int  main()
{
// 	for (size_t i = 0; i < 800; i++)
// 	{
// 		for (size_t j = 0; j < 600; j++)
// 		{
// 			frame_buffer[i][j].r = 20;
// 			frame_buffer[i][j].g = 0;
// 			frame_buffer[i][j].b = 100;
// 			z_buffer[i][j] = 0.0f;
// 		}
// 	}
	screen_init(800, 600, _T("Cube"));
	BindFB(800, 600, screen_fb);
	SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
// 	screen_fb = frame_buffer;
// 		for (size_t i = 300; i < 600; i++)
// 		{
// 			for (size_t j = 200; j <320; j++)
// 			{
// 				SetPixel(GetCDC(), i,j, RGB(frame_buffer[i][j].r, 0, 0));
// 			}
// 		}

// 		screen_fb[(int)cube_processed[i].pos.x * 800 + (int)cube_processed[i].pos.y] = 0;
// 		SetPixel(GetCDC(), cube_processed[i].pos.x, cube_processed[i].pos.y, RGB(255,255,255));
// 		framebuffer[(int)cube_processed[i].pos.x][(int)cube_processed[i].pos.y] = 0;
// 		screen_update();
// 		getchar();
// 		return 0;

	SetTimer(NULL,1, 1000/15, timerProc);
	clock_t start, finish;
	float  fps;
	/* 测量一个事件持续的时间*/
	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		start = clock();
		if (screen_keys[VK_LEFT])
		{
			cam_angle += 0.01f;
			RotateCube(cam_angle);
			//		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
			ClearFrameBuffer();
		}
		if (screen_keys[VK_RIGHT])
		{
			cam_angle -= 0.01f;
			RotateCube(cam_angle);
//		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
			ClearFrameBuffer();
		}

	DrawCube(self_angle,cam_angle);
//	for (size_t i = 0; i < 8; i++)
		//printf("x: %f y: %f z: %f w: %f\n",cube_processed[i].pos.x, cube_processed[i].pos.y, cube_processed[i].pos.z, cube_processed[i].pos.w);
	finish = clock();
	fps = 1.0f/((float)(finish - start) / CLOCKS_PER_SEC);
	if (fps<60.0f)
	{
		printf("FPS: %f\n", fps);
	}

		screen_update();
//		Sleep(1);
	}
	// 	continue;

	return 0;
}