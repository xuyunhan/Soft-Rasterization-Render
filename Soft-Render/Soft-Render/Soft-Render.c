#include <stdio.h>
#include "Math_Lib.h"
#include "Show_Window.h"
#include <Windows.h>
#include <time.h>
typedef struct
{
	int r, g, b;
}Color;

typedef struct
{
	unsigned int r, g, b;
}Color_int;

typedef struct
{
	vector_t pos;
	Color color;
}Vertex;

Vertex cube[8] = 
{
	{ { 1, -1,  1, 1 },{ 1.0f, 0.2f, 0.2f } },
	{ { -1, -1,  1, 1 },{ 0.2f, 1.0f, 0.2f } },
	{ { -1,  1,  1, 1 },{ 0.2f, 0.2f, 1.0f } },
	{ { 1,  1,  1, 1 },{ 1.0f, 0.2f, 1.0f } },
	{ { 1, -1, -1, 1 },{ 1.0f, 1.0f, 0.2f } },
	{ { -1, -1, -1, 1 },{ 0.2f, 1.0f, 1.0f } },
	{ { -1,  1, -1, 1 },{ 1.0f, 0.3f, 0.3f } },
	{ { 1,  1, -1, 1 },{ 0.2f, 1.0f, 0.3f } }
};

Vertex cube_processed[8] =
{
	{ { 1, -1,  1, 1 },{ 1.0f, 0.2f, 0.2f } },
	{ { -1, -1,  1, 1 },{ 0.2f, 1.0f, 0.2f } },
	{ { -1,  1,  1, 1 },{ 0.2f, 0.2f, 1.0f } },
	{ { 1,  1,  1, 1 },{ 1.0f, 0.2f, 1.0f } },
	{ { 1, -1, -1, 1 },{ 1.0f, 1.0f, 0.2f } },
	{ { -1, -1, -1, 1 },{ 0.2f, 1.0f, 1.0f } },
	{ { -1,  1, -1, 1 },{ 1.0f, 0.3f, 0.3f } },
	{ { 1,  1, -1, 1 },{ 0.2f, 1.0f, 0.3f } }
};

// const int a = 800 * 600;
// Color_int frame_buffer[800][600];
// float z_buffer[800][600];

unsigned int **framebufferPtr;      // ���ػ��棺framebuffePtr[y] ����� y��
float **zbufferPtr;//��Ȼ���
float *zbuffer;
float zbufferMaxDepth[800][600];//����memcpy����zbuffer��
unsigned int texbuffer[256][256];

typedef struct {
	matrix_t world;         // ��������任
	matrix_t view;          // ��Ӱ������任
	matrix_t projection;    // ͶӰ�任
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // ��Ļ��С
}	TransformMatrix;

TransformMatrix transformMatrix;

void ApplyClipping()//�ü��任
{
	;
}
void ApplyHomogenize()//��һ����NDC
{
	float rhw = 0;
	for (size_t i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_processed[i].pos.w;
//		matrix_apply(&cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//������ռ�
		cube_processed[i].pos.x = (cube_processed[i].pos.x * rhw + 1.0f) * 800.0f * 0.5f;
		cube_processed[i].pos.y = (1.0f - cube_processed[i].pos.y * rhw) * 600.0f * 0.5f;
		cube_processed[i].pos.z = cube_processed[i].pos.z * rhw;
		cube_processed[i].pos.w = 1.0f;
	};

}

void ApplyWVPTransform()//����任
{
	for (size_t i = 0; i < 8; i++)
	{
		matrix_apply(&cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform );//��CVV�ռ�
	};
	//��͸�ӱ任�������CVV���ڵĿռ䣬����CVV�Ǹ������壬���Ժܷ���Ĳü�������CVV�Ǹ������壬���Գ������ʱ���仯����Ҫ��������
	ApplyHomogenize();//��һ����NDC
	ApplyClipping();//�ü�
}

void Barycentric(float x1,float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, int color)
{
	float a, b, c;//����ϵ��
//	ƽ�������Ax + By + Cz + D = 0����ʾ�����z = f(x, y) = (-Ax - By - D) / C��
//	���Թ۲쵽��f(x + 1, y) �C f(x, y) = -A / C��f(x, y + 1) �C f(x, y) = -B / C����˲������ÿ���㶼����zֵ��ֻҪ����������һ�������zֵ�Ϳ����üӼ�������������zֵ��
	float A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
	float B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
	float C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
	float D = -(A*x1 + B*y1 + C*z1);
	float deltaZperX = -A / C;//�ԣ�x1,y1,z1��Ϊ��ʼ��
	float deltaZperY = -B / C;
	float z = 0;

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
//				printf("�㣺x: %d y: %d\n", x, y);
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//��������ε�Zֵ
				if (z < (zbuffer[(int)x * 800 + (int)y]))//������Ļ��������ʾ��������
				{
					zbuffer[(int)x*800+(int)y] = z;
					framebufferPtr[x][y] = color;// SetPixel(GetCDC(), x, y, RGB(255, 255, 255));
				}
			}
		}
	}

// 	getchar();
}
void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3,int color)//����ͼԪ
{
// 	ApplyWorldTransform();//��������Ӿֲ�����ϵת����������ϵ
// 	ApplyViewTransform();//���������ϵ
// 	ApplyProjectionTransform();//��CVV����ϵ��͸�ӳ�����ͶӰ�ռ䣩
// 	ApplyHomogenize();//��һ����NDC

	//�����޳�
//	double z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x);
	if ((p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x) < 0.0f)
	{
		//��ʼ��դ��
		Barycentric(p1->pos.x, p1->pos.y, p1->pos.z, p2->pos.x, p2->pos.y, p2->pos.z,p3->pos.x, p3->pos.y, p3->pos.z, color);
	}
}

void DrawPlane(int a,int b,int c, int d,int color)//�����ı��Σ��ĸ�����Ϊ���������
{
	Vertex* p1 = &cube_processed[a], *p2 = &cube_processed[b], *p3 = &cube_processed[c], *p4 =& cube_processed[d];

	DrawPrimitive(p1, p2, p3, color);//�ֳ�������ͼԪ����
	DrawPrimitive(p3, p4, p1, color);
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[a].pos.x, cube_processed[a].pos.y, cube_processed[a].pos.z, cube_processed[a].pos.w);
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[b].pos.x, cube_processed[b].pos.y, cube_processed[b].pos.z, cube_processed[b].pos.w);
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[c].pos.x, cube_processed[c].pos.y, cube_processed[c].pos.z, cube_processed[c].pos.w);
// 	printf("\n");
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[c].pos.x, cube_processed[c].pos.y, cube_processed[c].pos.z, cube_processed[c].pos.w);
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[d].pos.x, cube_processed[d].pos.y, cube_processed[d].pos.z, cube_processed[d].pos.w);
// 	printf("��������Ļ���꣺x: %f y: %f z: %f w: %f\n", cube_processed[a].pos.x, cube_processed[a].pos.y, cube_processed[a].pos.z, cube_processed[a].pos.w);
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
	ApplyWVPTransform();//��������Ӿֲ�����ϵת���ü����NDC
	//��ʼ��դ��
	DrawPlane(2, 3, 0, 1, 100);
	DrawPlane(7, 6, 5, 4, -250);
	DrawPlane(0, 4, 5, 1, -50);
	DrawPlane(1, 5, 6, 2, 200);
	DrawPlane(3, 2, 6, 7, -550);
	DrawPlane(3, 7, 4, 0, 0);
}

void RotateCube(float angle)
{
	matrix_t m;
	matrix_set_rotate(&m, -1, -0.5,1, angle);//����ת����
	transformMatrix.world = m;
	UpdateMVPMatrix();//���¾���
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
	int j;
	for (size_t i = 0; i < 800; i++)
	{
		for (size_t j = 0; j < 600; j++)
		{
			zbufferMaxDepth[i][j] = 1.0f;
		}
	}
// 	assert(ptr);
	framebufferPtr = (unsigned int **)ptr;//ԭ����
//	framebuffer = (unsigned int **)malloc(sizeof(unsigned int*) * height);
	for (j = 0; j < height; j++)
	{
//		framebuffer[j] = (unsigned int *)malloc(sizeof(void*));//framebuffer��ÿһ��ָ��fb��ÿһ�ж�Ӧ�ĵ�ַ
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
		framebuf = (unsigned int*)fb;//framebufָ��fb���׵�ַ
	for (j = 0; j < height; j++)
	{
		framebufferPtr[j] = (unsigned int *)(framebuf + width * 4 * j);//framebuffer��ÿһ��ָ��fb��ÿһ�ж�Ӧ�ĵ�ַ
 		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
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
	SetTimer(NULL,1, 1000/30, timerProc);
	clock_t start, finish;
	float  fps;
	/* ����һ���¼�������ʱ��*/
	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		start = clock();
		if (screen_keys[VK_LEFT])
		{
			cam_angle += 0.01f;
			RotateCube(cam_angle);
	//		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
			ClearFrameBuffer();
		}

	DrawCube(self_angle,cam_angle);
//	for (size_t i = 0; i < 8; i++)
		//printf("x: %f y: %f z: %f w: %f\n",cube_processed[i].pos.x, cube_processed[i].pos.y, cube_processed[i].pos.z, cube_processed[i].pos.w);
	finish = clock();
	fps = 1.0f/((float)(finish - start) / CLOCKS_PER_SEC);
	printf("FPS: %f\n", fps);

		screen_update();
//		Sleep(1);
	}
	// 	continue;

	return 0;
}