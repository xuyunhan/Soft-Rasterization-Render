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

typedef Pos Vector;//Pos��Vector��

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

unsigned int **framebufferPtr;      // ���ػ��棺framebuffePtr[y] ����� y��
float **zbufferPtr;//��Ȼ���
float *zbuffer;
float zbufferMaxDepth[800][600];//����memcpy����zbuffer��
unsigned int texbuffer[256][256];
float cube_camera_w[8];// ��һ��ǰ����ռ��ڶ����w���ݴ�

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
#pragma omp parallel for
	for (size_t i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_camera_w[i];
// 		cube_processed[i].rhw = rhw;
//		matrix_apply(& cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//������ռ�
		 cube_processed[i].pos.x = ( cube_processed[i].pos.x * rhw + 1.0f) * 800.0f * 0.5f;
		 cube_processed[i].pos.y = (1.0f -  cube_processed[i].pos.y * rhw) * 600.0f * 0.5f;
		 cube_processed[i].pos.z =  cube_processed[i].pos.z * rhw;
		 cube_processed[i].pos.w = 1.0f;
	};

}

void ApplyWVPTransform()//����任
{
#pragma omp parallel for
	for (size_t i = 0; i < 8; i++)
	{
		matrix_apply(&cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//��CVV�ռ�
	};
	ApplyClipping();//�ü�
}

unsigned int GetTexture(double u, double v)
{
// 	unsigned int nColorRef = RGB(0, 128, 255);//��ʱaΪBGR
// 	int red = nColorRef & 255;//������COLORREF�л�ȡRGB
// 	int green = nColorRef >> 8 & 255;
// 	int blue = nColorRef >> 16 & 255;
//	u *= 256.0f;
//	v *= 256.0f;
	int x = floor(u);
	int y = floor(v);
	double u_ratio = u - x;
	double v_ratio = v - y;
	double u_opposite = 1 - u_ratio;
	double v_opposite = 1 - v_ratio;

	unsigned int r = ((texbuffer[x][y] & 255) * u_opposite + (texbuffer[x + 1][y] & 255) * u_ratio) * v_opposite + ((texbuffer[x][y + 1] & 255) * u_opposite + (texbuffer[x + 1][y + 1] & 255) * u_ratio) * v_ratio;
	unsigned int g = ((texbuffer[x][y] >> 8 & 255) * u_opposite + (texbuffer[x + 1][y] >> 8 & 255) * u_ratio) * v_opposite + ((texbuffer[x][y + 1] >> 8 & 255) * u_opposite + (texbuffer[x + 1][y + 1] >> 8 & 255) * u_ratio) * v_ratio;
	unsigned int b = ((texbuffer[x][y] >> 16 & 255) * u_opposite + (texbuffer[x + 1][y] >> 16 & 255) * u_ratio) * v_opposite + ((texbuffer[x][y + 1] >> 16 & 255) * u_opposite + (texbuffer[x + 1][y + 1] >> 16 & 255) * u_ratio) * v_ratio;
//	return result;
//	return a;
	return RGB(r, g, b);

//	return texbuffer[(int)u][(int)v];
}

Vector dir_light = { -1,0,0,0 };//���ұ��յķ����

void Barycentric(Vertex* p1, Vertex* p2, Vertex* p3, Vector* normal)
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
	//����
	double u, v;
// 	float u1, v1, u2, v2, u3, v3;
	//����
	Vector _2N = { normal->x * 2,normal->y * 2,normal->z * 2,0 };
	Vector p = { p1->pos.x / 1600.0f +p3->pos.x / 1600.0f, p1->pos.y / 1200.0f+ p3->pos.y / 1200.0f, (p1->pos.z +p3->pos.z)/2.0f};
	Vector L;  vector_sub(&L, &dir_light, &p);
	Vector R,temp;
	temp = vector_multply(normal, vector_dotproduct(&_2N, &L));
	vector_sub(&R, &temp, &L);
	Vector V = { 0 - p.x,0 - p.y,0 - p.z,0 - p.w };
	float VdotR = vector_dotproduct(&V, &R);
	float VdotR_n = VdotR;
	int n = 1;//����
	VdotR_n = pow(VdotR, n);
//	for (size_t i = 0; i < n; i++)
	{
//		VdotR_n *= VdotR;
	}
	float Ks = 1; float Is = 1;
	float Ispecular = Ks*Is*VdotR_n;
	//��ñ����������ڵľ���
	float xmin, xmax, ymin, ymax;
	xmin = x1 < x2 ? x1 : x2;
	xmin = xmin < x3 ? xmin : x3;
	xmax = x1 > x2 ? x1 : x2;
	xmax = xmax > x3 ? xmax : x3;
	ymin = y1 < y2 ? y1 : y2;
	ymin = ymin < y3 ? ymin : y3;
	ymax = y1 > y2 ? y1 : y2;
	ymax = ymax > y3 ? ymax : y3;
// #pragma omp parallel for
#pragma omp parallel
{
#pragma omp parallel for nowait
	for (int x = xmin; x < xmax; x++)
	{
		for (int y = ymin; y < ymax; y++)
		{
			c = ((y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			b = ((y1 - y3)*x + (x3 - x1)*y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			a = 1 - b - c;
			if (a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)
			{
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//��������ε�Zֵ
				if (z < (zbuffer[(int)x * 800 + (int)y]))//������Ļ��������ʾ��������
				{
					double zr = a*(1 / p1->pos.w) + b*(1 / p2->pos.w) + c*(1 / p3->pos.w);
					u = ((a*(p1->texcoord.u / p1->pos.w) + b*(p3->texcoord.u / p2->pos.w) + c*(p3->texcoord.u / p3->pos.w)) / zr) * 255.0;//w
					v = ((a*(p1->texcoord.v / p1->pos.w) + b*(p2->texcoord.v / p2->pos.w) + c*(p3->texcoord.v / p3->pos.w)) / zr) * 255.0;//h
					zbuffer[(int)x * 800 + (int)y] = z;

// 					unsigned int color16 = 0xff2233;// GetTexture(u, v);
// 					byte r = 255, g = 0, b = 0;
// 					memcpy(&r, (char*)&color16 + 2, 1);
// 					memcpy(&g, (char*)&color16 + 1, 1);
// 					memcpy(&b, (char*)&color16, 1);
// //					r = r >> 1; g >> 2, b >> 2;
// 					int r10 = 0;// = r & 0x000001; r10 + ((r & 0x000010) >> 1) * 10;
// 					if (r & 0x000001)
// 					{
// 						r10 = 1;
// 					}
// 					if (r & 0x000010)
// 					{
// 						r10 += 10;
// 					}
// 					memcpy((char*)&color16 + 2,&r,  1);
// 					memcpy((char*)&color16 + 1, &g, 1);
// 					memcpy((char*)&color16,&b,  1);
// 					char a = r*2;
// 					framebufferPtr[x][y] = color16;
					framebufferPtr[x][y] = GetTexture(u, v);// *0x0000c0;
				}
			}
		}
	}
}

// 	getchar();
}

void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3)//����ͼԪ
{
// 	ApplyWorldTransform();//��������Ӿֲ�����ϵת����������ϵ
// 	ApplyViewTransform();//���������ϵ
// 	ApplyProjectionTransform();//��CVV����ϵ��͸�ӳ�����ͶӰ�ռ䣩
// 	ApplyHomogenize();//��һ����NDC

	//�����޳�
//	double z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p2->pos.y - p1->pos.y) * (p3->pos.x - p1->pos.x);
// 	��֪�����������η�����(a,b,c)
// 		a = ((p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y));
// 
// 		b = ((p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z));
// 
// 		c = ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));	
	Vector normal = { 0,0,0,0 };
	normal.x = (p2->pos.y - p1->pos.y) * (p3->pos.z - p1->pos.z) - (p3->pos.y - p1->pos.y) * (p2->pos.z - p1->pos.z);
	normal.y = (p2->pos.z - p1->pos.z) * (p3->pos.x - p1->pos.x) - (p3->pos.z - p1->pos.z) * (p2->pos.x - p1->pos.x);
	normal.z = (p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p3->pos.x - p1->pos.x) * (p2->pos.y - p1->pos.y);
	vector_normalize(&normal);//����ģ���﷨��Ҫ�淶��
	if (normal.z > 0.0f)
	{
		//��ʼ��դ��
		Barycentric(p1, p2, p3, &normal);
	}
}

void DrawPlane(int a,int b,int c, int d,int color)//�����ı��Σ��ĸ�����Ϊ���������
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

	DrawPrimitive(p3, p2, p1);//�ֳ�������ͼԪ����
	DrawPrimitive(p1, p4, p3);
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

int FPS = 0;
void DrawCube(float self_angle)
{
	ApplyWVPTransform();//��������Ӿֲ�����ϵת���ü����NDC
	for (size_t i = 0; i < 8; i++)
	{
		cube_camera_w[i] = cube_processed[i].pos.w;
	};
	//��͸�ӱ任�������CVV���ڵĿռ䣬����CVV�Ǹ������壬���Ժܷ���Ĳü�������CVV�Ǹ������壬���Գ������ʱ���仯����Ҫ��������
	ApplyHomogenize();//��һ����NDC
	//��ʼ��դ��

	DrawPlane(2, 3, 0, 1, 100);//ԭ����
	DrawPlane(7, 6, 5, 4, -250);
	DrawPlane(0, 4, 5, 1, -50);
	DrawPlane(1, 5, 6, 2, 200);
	DrawPlane(3, 2, 6, 7, -550);
	DrawPlane(3, 7, 4, 0, 0);
	FPS++;

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
	matrix_set_rotate(&m, -1, -0.5,1, angle);//����ת����
	transformMatrix.world = m;
	UpdateMVPMatrix();//���¾���
}

void SetCameraLookAt(TransformMatrix *transformMatrix, float x, float y, float z) {
	point_t eye = { x, y, z, 1 }, at = { 0, -0.5, 0.5, 1 }, up = { 0, 0, 1, 1 };
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
// 	framebufferPtr = (unsigned int **)ptr;//ԭ����
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
		framebuf = (char*)fb;//framebufָ��fb���׵�ַ
	for (j = 0; j < height; j++)
	{
		framebufferPtr[j] = (unsigned char *)(framebuf + width * 4 * j);//framebuffer��ÿһ��ָ��fb��ÿһ�ж�Ӧ�ĵ�ַ
 		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
	}

	//��ͼ����
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;
			texbuffer[j][i] = ((x + y) & 1) ? 0xffffff : RGB(255,0,0);
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
	printf("FPS: %d\n", FPS);
	FPS = 0;
// 	self_angle += 0.1f;
// 	ClearFrameBuffer();
// 	RotateCube(self_angle);
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
	static float x = 3.2, y = 0, z = 0;
	SetCameraLookAt(&transformMatrix, 3, 0, -0);
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

	SetTimer(NULL,1, 1000/1, timerProc);
//	clock_t start, finish;
//	float  fps;
	/* ����һ���¼�������ʱ��*/
	while (1/*screen_exit == 0 && screen_keys[VK_ESCAPE] == 0*/) {
// 		start = clock();
// 		if (screen_keys[VK_LEFT])
		{
			self_angle += 0.01f;
			RotateCube(self_angle);
//		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
			ClearFrameBuffer();
		}
// 		if (screen_keys[VK_RIGHT])
// 		{
// 			self_angle -= 0.1f;
// 			RotateCube(self_angle);
// //		SetCameraLookAt(&transformMatrix, 3.5, 0, cam_angle);
// 			ClearFrameBuffer();
// 		}

		DrawCube(self_angle);
//	for (size_t i = 0; i < 8; i++)
// 		printf("x: %f y: %f z: %f w: %f\n",cube_processed[i].pos.x, cube_processed[i].pos.y, cube_processed[i].pos.z, cube_processed[i].pos.w);
//	finish = clock();
//	fps = 1.0f/((float)(finish - start) / CLOCKS_PER_SEC);
// 	if (fps>110.0f)
//	{
// 		printf("FPS: %d\n", FPS);
//	}

		screen_update();
//  	getchar();
// 		Sleep(100);
	}
	// 	continue;

	return 0;
}