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

Vertex cube_NDC_space[8] =
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

Pos cube_world_space[8];
Pos cube_camera_space[8];
float eye_x = 3, y = 0, z = 0;

unsigned int **framebufferPtr;      // ���ػ��棺framebuffePtr[y] ����� y��
float **zbufferPtr;//��Ȼ���
float *zbuffer;
float zbufferMaxDepth[800][600];//����memcpy����zbuffer��
Color_float *texbuffer;
Color_float texbuffer1[256][256];
Color_float **texbufferPtr;
float cube_camera_w[8];// ��һ��ǰ����ռ��ڶ����w���ݴ�

typedef struct {
	matrix_t world;         // ��������任
	matrix_t view;          // ��Ӱ������任
	matrix_t projection;    // ͶӰ�任
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // ��Ļ��С
}	TransformMatrix;

TransformMatrix transformMatrix;

void ApplyHomogenize()//͸�ӱ任��CVV������һ����NDC
{
	float rhw = 0;
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_camera_w[i];
// 		cube_processed[i].rhw = rhw;
//		matrix_apply(& cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//������ռ�
		cube_NDC_space[i].pos.x = (cube_camera_space[i].x * rhw + 1.0f) * 800.0f * 0.5f;
		cube_NDC_space[i].pos.y = (1.0f - cube_camera_space[i].y * rhw) * 600.0f * 0.5f;
		cube_NDC_space[i].pos.z = cube_camera_space[i].z * rhw;
		 cube_NDC_space[i].pos.w = cube_camera_w[i];
	};

}

void ApplyWVPTransform()//�任��cameraspace
{
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		matrix_apply(&cube_camera_space[i], &cube[i].pos, &transformMatrix.transform);//��camera�ռ�
	};
}
Color_float *result = NULL;
void GetTexture(float u, float v)
{ 
// 	unsigned int nColorRef = RGB(0, 128, 255);//��ʱaΪBGR
// 	int red = nColorRef & 255;//������COLORREF�л�ȡRGB
// 	int green = nColorRef >> 8 & 255;
// 	int blue = nColorRef >> 16 & 255;
	if (u<0||v<0||u>254||v>254)
	{
		Color_float d = { 1, 0, 0 };
		return d;
	}
#if 1 //�������Բ�ֵ
// 	u *= 256.0f;
// 	v *= 256.0f;
	int x = u;
	int y = v;
	float u_ratio = u - x;
	float v_ratio = v - y;
	float u_opposite = 1 - u_ratio;
	float v_opposite = 1 - v_ratio;
// 	Color test = { 255, 126, 33 };
	 result->r = ((texbufferPtr[x][y].r) * u_opposite + (texbufferPtr[x + 1][y].r) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].r) * u_opposite + (texbufferPtr[x + 1][y + 1].r) * u_ratio) * v_ratio;
	 result->g = ((texbufferPtr[x][y].g) * u_opposite + (texbufferPtr[x + 1][y].g) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].g) * u_opposite + (texbufferPtr[x + 1][y + 1].g) * u_ratio) * v_ratio;
	 result->b = ((texbufferPtr[x][y].b) * u_opposite + (texbufferPtr[x + 1][y].b) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].b) * u_opposite + (texbufferPtr[x + 1][y + 1].b) * u_ratio) * v_ratio;
// 	int r = ((test.r) * u_opposite + (test.r) * u_ratio) * v_opposite + ((test.r) * u_opposite + (test.r) * u_ratio) * v_ratio;
// 	int g = ((test.g) * u_opposite + (test.g) * u_ratio) * v_opposite + ((test.g) * u_opposite + (test.g) * u_ratio) * v_ratio;
// 	int b = ((test.b) * u_opposite + (test.b) * u_ratio) * v_opposite + ((test.b) * u_opposite + (test.b) * u_ratio) * v_ratio;
// 	result = { r, g, b };
	return ;

#endif // 0
#if 0 //�޲�ֵ
	result->r = texbufferPtr[(int)u][(int)v].r;
	result->g = texbufferPtr[(int)u][(int)v].g;
	result->b = texbufferPtr[(int)u][(int)v].b;
	return;
#endif//	return a;

}

void NDCToCameraSpace(Pos *dst, const Pos* src)
{
// 	float rhw = 1.0f / src->w;
// 	dst->x = (src->x / 400.0f - 1.0f) / rhw;
	dst->x = (src->x / 400.0f - 1.0f) * src->w;
// 	dst->y = (1 - src->y / 300.0f) / rhw;
	dst->y = (1 - src->y / 300.0f) * src->w;
// 	dst->z = src->z / rhw;
	dst->z = src->z* src->w;
	dst->w = src->w;
	return;
}

Vector L;
Pos point_light_camera;
Pos t1;

void DrawFlatBottomTriangle(Pos * pointTop, Pos * pointLeft, Pos * pointRight, float deltaZperX, float deltaZperY, Vector *triangleNormal_rotated_camera_normalized, Vertex *p1, Vertex* p2, Vertex* p3)
{
	int xtop = pointTop->x, xleft = pointLeft->x, xright = pointRight->x, ytop = pointTop->y, ybtm = pointLeft->y;//ƽ�������Σ�pointLeft��pointRight��y����ybtm


	int dx1 = xtop - xleft;
	int dy1 = ybtm - ytop;
	int dx2 = xtop - xright;
	int dy2 = ybtm - ytop;
	//test
#pragma region ����
	float x1 = p1->pos.x;
	float y1 = p1->pos.y;
	float z1 = p1->pos.z;
	float x2 = p2->pos.x;
	float y2 = p2->pos.y;
	float z2 = p2->pos.z;
	float x3 = p3->pos.x;
	float y3 = p3->pos.y;
	float z3 = p3->pos.z;
	const float p1w = p1->pos.w, p2w = p2->pos.w, p3w = p3->pos.w;
	const float p1u = p1->texcoord.u, p2u = p2->texcoord.u, p3u = p3->texcoord.u;
	const float p1v = p1->texcoord.v, p2v = p2->texcoord.v, p3v = p3->texcoord.v;
	const float invp1w = 1.0f / p1w, invp2w = 1.0f / p2w, invp3w = 1.0f / p3w;

#pragma endregion 

	int xstep1 = 1, xstep2 = 1;
	if (dy1 <= 0 || dy2 <= 0)
		return;
	if (dx1 < 0)
	{
		xstep1 = -1;
		dx1 = -dx1;
	}
	if (dx2 < 0)
	{
		xstep2 = -1;
		dx2 = -dx2;
	}

	int error1 = 2 * dx1 - dy1, error2 = 2 * dx2 - dy2;
	int x, y;
	Pos eye = { eye_x, 0, 0, 0 };

	Pos p_currentDrawing_camera;
	if (xleft < 0) xleft = 0;
	if (xright>799) xright = 799;
	if (ytop < 1) ytop = 1;
	if (ybtm > 599)ybtm = 599;
	int x1tmp = xleft, x2tmp = xright;
	for (y = ybtm; y > ytop; y--)
	{
		while (error1 > 0)
		{
			x1tmp += xstep1;
			error1 -= 2 * dy1;
		}
		while (error2 > 0)
		{
			x2tmp += xstep2;
			error2 -= 2 * dy2;
		}

		int dx = x2tmp - x1tmp;
#pragma omp parallel for private(x) num_threads(4)
		for (x = x1tmp; x <= x2tmp; x++)
		{
			float px = (float)x;
			float c = ((y1 - y2)*px + (x2 - x1)*(float)y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			float b = ((y1 - y3)*px + (x3 - x1)*(float)y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			float a = 1 - b - c;
			float z = z1 + (px - x1) *deltaZperX + ((float)y - y1)*deltaZperY;//��������ε�Zֵ
// 			if (z < (zbuffer[(int)y * 800 + (int)x]))//������Ļ��������ʾ��������
			{
				float invzr = 1.0f / (a*(invp1w)+b*(invp2w)+c*(invp3w));
				float u = ((a*(p1u *invp1w) + b*(p3u *invp2w) + c*(p3u *invp3w)) *invzr) * 254.0f;//w
				float v = ((a*(p1v *invp1w) + b*(p2v *invp2w) + c*(p3v *invp3w)) *invzr) * 254.0f;//h
// 				zbuffer[(int)y * 800 + (int)x] = z;
				//�������ع���
				Pos temp = { x, y, z, invzr };
				NDCToCameraSpace(&p_currentDrawing_camera, &temp);
				vector_sub(&L, &point_light_camera, &p_currentDrawing_camera);
				L.w = 1;
				vector_normalize(&L);
				float diffuse2 = vector_dotproduct(&L, triangleNormal_rotated_camera_normalized);
				diffuse2 = max(diffuse2, 0);
// 					diffuse2 = min(diffuse2, 0.9f);
				//�����ع�����
				float specular = 0;
#if 0
				//Phongģ�����ع���
				if (diffuse2 < 1.0f)
				{
					Vector V; vector_sub(&V, &eye, &p_currentDrawing_camera);
					vector_normalize(&V);
					Vector t = vector_multply(&triangleNormal_rotated_camera_normalized, 2.0f*vector_dotproduct(&L, &triangleNormal_rotated_camera_normalized));//t = 2 * dot(N, L) * N
					Vector R; vector_sub(&R, &t, &L);//�����Lǰ��������Ŀ�������
					vector_normalize(&R);
// 					float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
					specular = powf(max(0, vector_dotproduct(&R, &V)), 2);
				}
				//Phongģ�����ع�����
#endif				
				GetTexture(u, v);
				float r = result->r * diffuse2 + specular;
				float g = result->g * diffuse2 + specular;
				float b = result->b * diffuse2 + specular;
// 				r = min(r, 1); g = min(g, 1); b = min(b, 1);

				framebufferPtr[y][x] = RGB(r*255.0f, g*255.0f, b*255.0f);
			}
		}

		error1 += 2 * dx1;
		error2 += 2 * dx2;
	}
}

void DrawFlatTopTriangle(Pos * pointBottom, Pos * pointLeft, Pos * pointRight, float deltaZperX, float deltaZperY, Vector *triangleNormal_rotated_camera_normalized, Vertex *p1, Vertex* p2, Vertex* p3)
{
	int xbottom = pointBottom->x, xleft = pointLeft->x, xright = pointRight->x, ytop = pointBottom->y, ybtm = pointLeft->y;//ƽ�������Σ�pointLeft��pointRight��y����ybtm
	int dx1 = xbottom - xleft;
	int dy1 = ytop - ybtm;
	int dx2 = xbottom - xright;
	int dy2 = ytop - ybtm;
	int xstep1 = 1, xstep2 = 1;

	//test
#pragma region ����
	float x1 = p1->pos.x;
	float y1 = p1->pos.y;
	float z1 = p1->pos.z;
	float x2 = p2->pos.x;
	float y2 = p2->pos.y;
	float z2 = p2->pos.z;
	float x3 = p3->pos.x;
	float y3 = p3->pos.y;
	float z3 = p3->pos.z;
	const float p1w = p1->pos.w, p2w = p2->pos.w, p3w = p3->pos.w;
	const float p1u = p1->texcoord.u, p2u = p2->texcoord.u, p3u = p3->texcoord.u;
	const float p1v = p1->texcoord.v, p2v = p2->texcoord.v, p3v = p3->texcoord.v;
	const float invp1w = 1.0f / p1w, invp2w = 1.0f / p2w, invp3w = 1.0f / p3w;

#pragma endregion 

	if (dy1 <= 0 || dy2 <= 0)
		return;
	if (dx1 < 0)
	{
		xstep1 = -1;
		dx1 = -dx1;
	}
	if (dx2 < 0)
	{
		xstep2 = -1;
		dx2 = -dx2;
	}

	int error1 = 2 * dx1 - dy1, error2 = 2 * dx2 - dy2;
	Pos p_currentDrawing_camera;
	int x, y;
	Pos eye = { eye_x, 0, 0, 0 };

	if (xleft < 1) xleft = 1;
	if (xright>799) xright = 799;
	if (ybtm < 0) ybtm = 0;
	if (ytop > 599)ytop = 599;
	int x1tmp = xleft, x2tmp = xright;

	for ( y= ybtm; y < ytop; y++)
	{
		while (error1 > 0)
		{
			x1tmp += xstep1;
			error1 -= 2 * dy1;
		}
		while (error2 > 0)
		{
			x2tmp += xstep2;
			error2 -= 2 * dy2;
		}

		int dx = x2tmp - x1tmp;

#pragma omp parallel for private(x) num_threads(4)
		for ( x = x1tmp; x <= x2tmp; x++)
		{
			float px = (float)x;
			float c = ((y1 - y2)*px + (x2 - x1)*(float)y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			float b = ((y1 - y3)*px + (x3 - x1)*(float)y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			float a = 1 - b - c;
			float z = z1 + (px - x1) *deltaZperX + ((float)y - y1)*deltaZperY;//��������ε�Zֵ
// 				if (z < (zbuffer[(int)y * 800 + (int)x]))//������Ļ��������ʾ��������
			{
				float invzr = 1.0f / (a*(invp1w)+b*(invp2w)+c*(invp3w));
				float u = ((a*(p1u *invp1w) + b*(p3u *invp2w) + c*(p3u *invp3w)) *invzr) * 254.0f;//w
				float v = ((a*(p1v *invp1w) + b*(p2v *invp2w) + c*(p3v *invp3w)) *invzr) * 254.0f;//h
// 					zbuffer[(int)y * 800 + (int)x] = z;
				//�������ع���
				Pos temp = { x, y, z, invzr };
				NDCToCameraSpace(&p_currentDrawing_camera, &temp);
				vector_sub(&L, &point_light_camera, &p_currentDrawing_camera) ;
				L.w = 1;
				vector_normalize(&L);
				float diffuse2 = vector_dotproduct(&L, triangleNormal_rotated_camera_normalized);
				diffuse2 = max(diffuse2, 0);
// 					diffuse2 = min(diffuse2, 0.9f);
				//�����ع�����
				float specular = 0;
#if 0
				//Phongģ�����ع���
				if (diffuse2 < 1.0f)
				{
					Vector V; vector_sub(&V, &eye, &p_currentDrawing_camera);
					vector_normalize(&V);
					Vector t = vector_multply(&triangleNormal_rotated_camera_normalized, 2.0f*vector_dotproduct(&L, &triangleNormal_rotated_camera_normalized));//t = 2 * dot(N, L) * N
					Vector R; vector_sub(&R, &t, &L);//�����Lǰ��������Ŀ�������
					vector_normalize(&R);
// 					float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
					specular = powf(max(0, vector_dotproduct(&R, &V)), 2);
				}
				//Phongģ�����ع�����
#endif				
				GetTexture(u, v);
				float r = result->r * diffuse2 + specular;
				float g = result->g * diffuse2 + specular;
				float b = result->b * diffuse2 + specular;
// 				r = min(r, 1); g = min(g, 1); b = min(b, 1);

				framebufferPtr[y][x] = RGB(r*255.0f, g*255.0f, b*255.0f);
			}
// 			drawPixel(pVram, x, y, RGBA(r, g, b, 0));
		}

		error1 += 2 * dx1;
		error2 += 2 * dx2;
	}
}

void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3, Vector* triangleNormal_rotated_camera_normalized, Vector* p1_camera, Vector* p2_camera, Vector* p3_camera)//����ͼԪ
{
	if ((p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p3->pos.x - p1->pos.x) * (p2->pos.y - p1->pos.y) < 0.0f)// ��������
		return;
	int i = 0, j = 0 ,x= 0,y=0;
#pragma region ����
	float x1 = p1->pos.x;
	float y1 = p1->pos.y;
	float z1 = p1->pos.z;
	float x2 = p2->pos.x;
	float y2 = p2->pos.y;
	float z2 = p2->pos.z;
	float x3 = p3->pos.x;
	float y3 = p3->pos.y;
	float z3 = p3->pos.z;

#pragma endregion 
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
	float u, v;

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

	int red = 0, green = 0, blue = 0;
	const int xend = xmax < 800 ? xmax : 800;
	const int yend = ymax < 600 ? ymax : 600;
	const int xstart = xmin > 0 ? xmin : 0;
	const int ystart = ymin > 0 ? ymin : 0;
	float zr = 0, invzr = 1;
	Color_float texColor;
	const float p1w = p1->pos.w, p2w = p2->pos.w, p3w = p3->pos.w;
	const float p1u = p1->texcoord.u, p2u = p2->texcoord.u, p3u = p3->texcoord.u;
	const float p1v = p1->texcoord.v, p2v = p2->texcoord.v, p3v = p3->texcoord.v;
	const float invp1w = 1.0f / p1w, invp2w = 1.0f / p2w, invp3w = 1.0f / p3w;
	//�����ع���
	float diffuse2;
// 	Pos p;// = { (op1->x + op2->x + op3->x + op4->x)*0.25f, (op1->y + op2->y + op3->y + op4->y)*0.25f, (op1->z + op2->z + op3->z + op4->z)*0.25f, 1 };
	Pos p_currentDrawing_camera;
#pragma region Phongģ��
// 	Phongģ����Ϊ���淴��Ĺ�ǿ�뷴����ߺ����ߵļн���أ�
// 		Ispec = Ks * Il * (dot(V, R)) ^ Ns
// 		����Ks Ϊ���淴��ϵ��,Il�ǹ�Դǿ�ȣ�Ns�Ǹ߹�ָ����V��ʾ�Ӷ��㵽�ӵ�Ĺ۲췽��R������ⷽ�����ڷ����ķ���R����ͨ������ⷽ��L(�Ӷ���ָ���Դ)������ķ����������
// 		R + L = 2 * dot(N, L) * N  �� R = 2 * dot(N, L) * N - L
// 		�������յļ���ʽΪ��
// 		Ispec = Ks * Il * (dot(V, (2 * dot(N, L) * N �C L)) ^ Ns
	float Ks = 0.7f, Il = 1.0f, Ns = 20.0f;
	Pos eye = { eye_x, 0, 0, 0 };
// 	Ispec = max(Ispec, 0);
// 		Ispec = min(Ispec, 1);
#pragma endregion 
	//�����ع�����

#if 0
#pragma region Barycentric�㷨����
// #pragma omp parallel for private(x,y) num_threads(1)
	for (x = xstart; x < xend; x++)
	{
		for (y = ystart; y < yend; y++)
		{
			c = ((y1 - y2)*x + (x2 - x1)*y + x1*y2 - x2*y1) / ((y1 - y2)*x3 + (x2 - x1)*y3 + x1*y2 - x2*y1);
			b = ((y1 - y3)*x + (x3 - x1)*y + x1*y3 - x3*y1) / ((y1 - y3)*x2 + (x3 - x1)*y2 + x1*y3 - x3*y1);
			a = 1 - b - c;
			if (a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)
			{
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//��������ε�Zֵ
// 				if (z < (zbuffer[(int)y * 800 + (int)x]))//������Ļ��������ʾ��������
				{
					zr = a*(invp1w)+b*(invp2w)+c*(invp3w);
					invzr = 1.0f / zr;// a*(invp1w)+b*(invp2w)+c*(invp3w);
					u = ((a*(p1u *invp1w) + b*(p3u *invp2w) + c*(p3u *invp3w)) *invzr) * 254.0f;//w
					v = ((a*(p1v *invp1w) + b*(p2v *invp2w) + c*(p3v *invp3w)) *invzr) * 254.0f;//h
// 					zbuffer[(int)y * 800 + (int)x] = z;
					//�������ع���
					Pos temp = { x, y, z , invzr};
					NDCToCameraSpace(&p_currentDrawing_camera, &temp);
					vector_sub(&L, &point_light_camera, &p_currentDrawing_camera);
					L.w = 1;
					vector_normalize(&L);
					diffuse2 = vector_dotproduct(&L, triangleNormal_rotated_camera_normalized);
					diffuse2 = max(diffuse2, 0);
// 					diffuse2 = min(diffuse2, 0.9f);
					//�����ع�����
					//Phongģ�����ع���
					float specular = 0;
					if (diffuse2 > 1.0f)
					{
						Vector V; vector_sub(&V, &eye, &p_currentDrawing_camera);
						vector_normalize(&V);
						Vector t = vector_multply(&triangleNormal_rotated_camera_normalized, 2.0f*vector_dotproduct(&L, &triangleNormal_rotated_camera_normalized));//t = 2 * dot(N, L) * N
						Vector R; vector_sub(&R, &t, &L);//�����Lǰ��������Ŀ�������
						vector_normalize(&R);
						float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
						specular = powf(max(0, vector_dotproduct(&R, &V)), 2);
					}
					//Phongģ�����ع�����
					GetTexture(u, v);
					float r = result->r * diffuse2 + specular;
					float g = result->g * diffuse2 + specular;
					float b = result->b * diffuse2 + specular;
// 					float r = result->r * diffuse2 ;
// 					float g = result->g * diffuse2 ;
// 					float b = result->b * diffuse2 ;
					r = min(r, 1); g = min(g, 1); b = min(b, 1);

					framebufferPtr[y][x] = RGB(r*255.0f, g*255.0f, b*255.0f);
				}
			}
		}
	}
#pragma endregion Barycentric�㷨����
#endif
#pragma region Bresenham�㷨
	Pos point1 = p1->pos, point2 = p2->pos, point3 = p3->pos;
	Pos *pointTop = &point1, *pointLeft = &point1, *pointRight = &point1;
	//��������һ�������Σ�����yֵ�ֳ������㣬ptop��pmid��pbottom��ͨ��xbottom+(xtop-xbottom)/(ybottom-ytop)*(ybottom-ymid)���������pmid��ĺ�������������һ�ߵĽ����x���꣬
	//�˽����pmid�����������η�Ϊƽ�����Ǻ�ƽ������
	Pos *pmid = &point1, *ptop = &point2 , *pbottom = &point3, *tmp;
	if (pbottom->y < pmid->y) { tmp = pmid; pmid = pbottom; pbottom = tmp; }//�������͵ڶ����Ƚϣ��ϸ�
	if (pmid->y < ptop->y) { tmp = pmid; pmid = ptop; ptop = tmp; }//�ڶ����͵�һ���Ƚϣ��ϸ�����ʱtop�Ѿ��ҵ�
	if (pbottom->y < pmid->y) { tmp = pmid; pmid = pbottom; pbottom = tmp; }//�������͵ڶ����Ƚϣ��ϸ����ҵ�mid��bottom
	Pos crsPoint = { pbottom->x + (ptop->x - pbottom->x) / (pbottom->y - ptop->y)*(pbottom->y - pmid->y), pmid->y, 0, 0 };
	if (pmid->x < crsPoint.x) { pointLeft = pmid; pointRight = &crsPoint; }//ȷ��left��right
	else { pointLeft = &crsPoint; pointRight = pmid; }

	DrawFlatBottomTriangle(ptop, pointLeft, pointRight, deltaZperX, deltaZperY, triangleNormal_rotated_camera_normalized, p1, p2, p3);
	DrawFlatTopTriangle(pbottom, pointLeft, pointRight, deltaZperX, deltaZperY, triangleNormal_rotated_camera_normalized, p1, p2, p3);

// 	if (pointTop->y > point2.y ) pointTop = &point2;
// 	if (pointTop->y > point3.y) pointTop = &point3;
// 	if (pointLeft->x >= point2.x && pointLeft->y <=point2.y) pointLeft = &point2;
// 	if (pointLeft->x >= point3.x && pointLeft->y <=point3.y) pointLeft = &point3;
// 	if (pointRight->x <= point2.x && pointRight->y <=point2.y) pointRight = &point2;
// 	if (pointRight->x <= point3.x && pointRight->y <=point3.y) pointRight = &point3;
	


	return;
#pragma endregion Bresenham�㷨

	// 	getchar();
}
	Pos point_light_world = { 1.8f, 0, 0, 1};

void DrawPlane(int a,int b,int c, int d)//�����ı��Σ��ĸ�����Ϊ���������
{
	Vertex* p1 = &cube_NDC_space[a], *p2 = &cube_NDC_space[b], *p3 = &cube_NDC_space[c], *p4 =& cube_NDC_space[d];
	p1->pos.w = cube_camera_w[a]; p2->pos.w = cube_camera_w[b]; p3->pos.w = cube_camera_w[c]; p4->pos.w = cube_camera_w[d];
// 	Vertex tp1, tp2, tp3,tp4;
// 	tp1.pos = p1->pos; tp2.pos = p2->pos; tp3.pos = p3->pos; tp4.pos = p4->pos;
// 	tp1.rhw = p1->rhw; tp2.rhw = p2->rhw; tp3.rhw = p3->rhw; tp4.rhw = p4->rhw;
// 	tp1.pos.w = tp2.pos.w = cube_camera_w[b]; tp3.pos.w = cube_camera_w[c]; tp4.pos.w = cube_camera_w[d];
	p1->texcoord.u = 0, p1->texcoord.v = 0, p2->texcoord.u = 0, p2->texcoord.v = 1;
	p3->texcoord.u = 1, p3->texcoord.v = 1, p4->texcoord.u = 1, p4->texcoord.v = 0;
// 	tp1.texcoord = p1->texcoord; tp2.texcoord = p2->texcoord; tp3.texcoord = p3->texcoord; tp4.texcoord = p4->texcoord;
	//������Ͱѹ����������...һ��ƽ��һ�����գ���ƽ�й���˵
	Vector normal_camera = { 0, 0, 0, 0 };
	Pos* op1 = &cube[a].pos, *op2 = &cube[b].pos, *op3 = &cube[c].pos, *op4 = &cube[d].pos;
	Pos* cp1 = &cube_camera_space[a], *cp2 = &cube_camera_space[b], *cp3 = &cube_camera_space[c], *cp4 = &cube_camera_space[d];

	// 	��֪�����������η�����(a,b,c)
// 		a = ((p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y));
// 		b = ((p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z));
// 		c = ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));	
	// 	//��������op�淨�����ǶԵģ�������Ĺ�ʽ���������ˣ������������worldspace�µģ����Բ�����
// 	normal.x = (op2->y - op3->y) * (op1->z - op3->z) - (op1->y - op3->y) * (op2->z - op3->z);
// 	normal.y = (op2->z - op3->z) * (op1->x - op3->x) - (op1->z - op3->z) * (op2->x - op3->x);
// 	normal.z = (op2->x - op3->x) * (op1->y - op3->y) - (op1->x - op3->x) * (op2->y - op3->y);
// 	//ֱ����p��cameraspace�µ�����
	normal_camera.x = (cp2->y - cp3->y) * (cp1->z - cp3->z) - (cp1->y - cp3->y) * (cp2->z - cp3->z);
	normal_camera.y = (cp2->z - cp3->z) * (cp1->x - cp3->x) - (cp1->z - cp3->z) * (cp2->x - cp3->x);
	normal_camera.z = (cp2->x - cp3->x) * (cp1->y - cp3->y) - (cp1->x - cp3->x) * (cp2->y - cp3->y);
	//�������diffuse������cameraspace����
	vector_normalize(&normal_camera);//����ģ���﷨��Ҫ�淶��
	matrix_apply(&t1, &point_light_world, &transformMatrix.view);
	matrix_apply(&point_light_camera, &t1, &transformMatrix.projection);

#if 0 // �����Ĵ���
	Vector L, normal_rotated, normal_rotated_camera;
	Vector p_rotated_world; Vector p_rotated_camera; Vector dir_light_camera, dir_light_NDC;
	Vector t1;
	matrix_apply(&t1, &dir_light_world, &transformMatrix.view);
	matrix_apply(&dir_light_camera, &t1, &transformMatrix.projection);
	//��p1��NDC�ص�cameraspace��������test_p_in_cameraspace
	Pos test_p_in_cameraspace;
	NDCToCameraSpace(&test_p_in_cameraspace, &p1->pos);
	//�ź���
	vector_sub(&L, &dir_light_camera, &test_p_in_cameraspace);

	vector_normalize(&L);
	float dir_light_diffuse = vector_dotproduct(&L, &normal_camera);
	dir_light_diffuse = max(dir_light_diffuse, 0);
// // 	diffuse = min(diffuse, 1);
#endif
// 	Vector dir_light_world = { 2,0,-0,0 };//����Ļ���յķ����

// 	matrix_apply(&normal_rotated, &normal, &transformMatrix.world);
// 	matrix_apply(&normal_rotated_camera, &normal_rotated, &transformMatrix.projection);
// 	matrix_apply(&normal_rotated_camera, &normal, &transformMatrix.transform);
// 	vector_normalize(&normal_rotated);//����ģ���﷨��Ҫ�淶��
// 	vector_normalize(&normal_rotated_camera);//����ģ���﷨��Ҫ�淶��
// 	Vector p = { (op1->x + op2->x + op3->x + op4->x)*0.25f, (op1->y + op2->y + op3->y + op4->y)*0.25f, (op1->z + op2->z + op3->z + op4->z)*0.25f, 1 };
// 	matrix_apply(&p_rotated_world, &p, &transformMatrix.world);
// 	matrix_apply(&p_rotated_camera, &p_rotated_world, &transformMatrix.projection);
// 	matrix_apply(&p_rotated_camera, &p, &transformMatrix.transform);
// 	vector_sub(&L, &dir_light_world, &p_rotated_world);
// 	Vector R,temp;
// 	temp = vector_multply(normal, vector_dotproduct(&_2N, &L));
// 	vector_sub(&R, &temp, &L);
// 	float diffuse = vector_dotproduct(&L, &normal_rotated);
	
// 	Vector p = { (cp1->x + cp2->x + cp3->x + cp4->x)*0.25f, (cp1->y + cp2->y + cp3->y + cp4->y)*0.25f, (cp1->z + cp2->z + cp3->z + cp4->z)*0.25f, 1 };
// 	Vector pNDC = { (p1->pos.x + p2->pos.x + p3->pos.x + p4->pos.x)*0.25f, (p1->pos.y + p2->pos.y + p3->pos.y + p4->pos.y)*0.25f, (p1->pos.z + p2->pos.z + p3->pos.z + p4->pos.z)*0.25f, 1 };
// 	Vector normal;
// 	normal.x = (p2->pos.y - p3->pos.y) * (p1->pos.z - p3->pos.z) - (p1->pos.y - p3->pos.y) * (p2->pos.z - p3->pos.z);
// 	normal.y = (p2->pos.z - p3->pos.z) * (p1->pos.x - p3->pos.x) - (p1->pos.z - p3->pos.z) * (p2->pos.x - p3->pos.x);
// 	normal.z = (p2->pos.x - p3->pos.x) * (p1->pos.y - p3->pos.y) - (p1->pos.x - p3->pos.x) * (p2->pos.y - p3->pos.y);
// 	normal.w = 0;
// 	vector_normalize(&normal);
// 	float rhw = 1.0f / dir_light_camera.w;
// 	dir_light_NDC.x = (dir_light_camera.x * rhw + 1.0f) * 800.0f * 0.5f;
// 	dir_light_NDC.y = (1.0f - dir_light_camera.y * rhw) * 600.0f * 0.5f;
// 	dir_light_NDC.z = dir_light_camera.z * rhw;
// 	dir_light_NDC.w = dir_light_camera.w;

// #pragma region Phongģ��
// // 	Phongģ����Ϊ���淴��Ĺ�ǿ�뷴����ߺ����ߵļн���أ�
// // 		Ispec = Ks * Il * (dot(V, R)) ^ Ns
// // 		����Ks Ϊ���淴��ϵ��,Il�ǹ�Դǿ�ȣ�Ns�Ǹ߹�ָ����V��ʾ�Ӷ��㵽�ӵ�Ĺ۲췽��R������ⷽ�����ڷ����ķ���R����ͨ������ⷽ��L(�Ӷ���ָ���Դ)������ķ����������
// // 		R + L = 2 * dot(N, L) * N  �� R = 2 * dot(N, L) * N - L
// // 		�������յļ���ʽΪ��
// // 		Ispec = Ks * Il * (dot(V, (2 * dot(N, L) * N �C L)) ^ Ns
// 	float Ks = 0.7f, Il = 1.0f,Ns = 20.0f;
// 	Vector V; vector_sub(&V, &dir_light_world, &p_rotated_world);
// 	vector_normalize(&V);
// 	Vector t = vector_multply(&normal_rotated, 2.0f*vector_dotproduct(&L, &normal_rotated));
// 	Vector R; vector_sub(&R, &t, &L);
// 	vector_normalize(&R);
// 	float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
// 	Ispec = max(Ispec, 0);
// // 		Ispec = min(Ispec, 1);
// #pragma endregion 

// 	Pos op1_world, op2_world, op3_world, op4_world;
// 	matrix_apply(&op1_world, op1, &transformMatrix.world);
// 	matrix_apply(&op2_world, op2, &transformMatrix.world);
// 	matrix_apply(&op3_world, op3, &transformMatrix.world);
// 	matrix_apply(&op4_world, op4, &transformMatrix.world);
	//�ֳ�������ͼԪ���ƣ���ʱ��Ϊ���Ʒ�ʽ
	DrawPrimitive(p3, p2, p1, &normal_camera, cp3, cp3, cp1);//����ȥ��cp��cameraspace�µĶ�������
	DrawPrimitive(p1, p4, p3, &normal_camera, cp1, cp4, cp3);
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
	ApplyWVPTransform();//��������Ӿֲ�����ϵת���ü����NDC
	for (size_t i = 0; i < 8; i++)
	{
		cube_camera_w[i] = cube_camera_space[i].w;
	}
	//��͸�ӱ任�������CVV���ڵĿռ䣬����CVV�Ǹ������壬���Ժܷ���Ĳü�������CVV�Ǹ������壬���Գ������ʱ���仯����Ҫ��������
	ApplyHomogenize();//��һ����NDC
	//��ʼ��դ��

	DrawPlane(2, 3, 0, 1);
	DrawPlane(0, 4, 5, 1);
	DrawPlane(1, 5, 6, 2);
	DrawPlane(3, 2, 6, 7);
	DrawPlane(7, 4, 0, 3);//һ��ʼ������Ļ
	DrawPlane(7, 6, 5, 4);//

	FPS++;
}

void RotateCube(float theta)//����Ϊ����
{
	matrix_t m;
// 	matrix_set_rotate(&m, 0, 1, 0, theta);//��y��ת
// 	matrix_set_rotate(&m, 1,0 , 0, theta);//��x��ת
	matrix_set_rotate(&m, 0, 0, 1, theta);//��z��ת
// 	matrix_set_rotate(&m, -1, -0.5, 1, theta);//����ת����
	transformMatrix.world = m;
	UpdateMVPMatrix();//���¾���
}

void SetCameraLookAt(TransformMatrix *transformMatrix, float x, float y, float z) {//todo::��������и����⣬lookat�������view����̫�ԣ����Թ̶����λ���ڣ�0,3,0����ֻ���ƣ�3,2,7,6��������ĳ�������������֣���MVP�任�󣬹�һ��֮ǰ��CVV�������ʹ���
	point_t eye = { x, y, z, 1 }, at = { 0, 0, 0, 1 }, up = { 0, 0, 1, 0 };//y����
	matrix_set_identity(&transformMatrix->world);
	matrix_set_identity(&transformMatrix->view);
	matrix_set_perspective(&transformMatrix->projection, 3.1415926f * 0.5f, 800.0f/600.0f, 1.0f, 300.0f);
	matrix_set_lookat(&transformMatrix->view, &eye, &at, &up);
	UpdateMVPMatrix();
}

void BindFB(int width, int height, void *fb) {
// 	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
// 	char *ptr = (char*)malloc(need + 64);
	char *framebuf, *zbuf, *texbuf;
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

	texbufferPtr = (Color_float **)malloc(sizeof(Color_float*)* 256);
	texbuffer = (Color_float *)malloc(256 * 256 * sizeof(Color_float));
//	memset(zbuffer, 1, 800 * 600 * sizeof(float));
	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
// 	ptr += sizeof(void*) * height * 2;
// 	device->texture = (IUINT32**)ptr;
// 	ptr += sizeof(void*) * 1024;
// 	framebuf = (char*)ptr;
 //	zbuf = (char*)ptr + width * height * 4;
	zbuf = (char*)zbufferPtr;
	texbuf = (char*)texbuffer;
// 	ptr += width * height * 8;
	//xuyunhan
//  	screen_fb = (char*)malloc(800 * 600 * 4);
// 	BITMAPINFO bi = { { sizeof(BITMAPINFOHEADER), 800, -600, 1, 32, BI_RGB,	800 * 600 * 4, 0, 0, 0, 0 } };
// 	screen_hb = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &screen_fb, 0, 0);

// 	memset(screen_fb, 1, 800 * 600 * 4);

// 	if (fb != NULL)
		framebuf = screen_fb;//framebufָ��fb���׵�ַ

	for (j = 0; j < 600; j++)
	{
		framebufferPtr[j] = (unsigned int *)(framebuf + 800 * 4 * j);//framebuffer��ÿһ��ָ��fb��ÿһ�ж�Ӧ�ĵ�ַ
		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
	}
	for (j = 0; j < 256; j++)
	{
		texbufferPtr[j] = (Color_float *)(texbuf + 256 * sizeof(Color_float) * j);//framebuffer��ÿһ��ָ��fb��ÿһ�ж�Ӧ�ĵ�ַ
	}

	//��ͼ����
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;

		Color_float white = { 1,1,1 }, blue = {0.9372f,0.7451f,0.247f }, blue2 = { 239,0,0 };
//		texbuffer[j][i] = ((x + y) & 1) ? 0xffffff : RGB(255,0,0);
		texbuffer1[j][i] = ((x + y) & 1) ? white : blue;
		}
	}
	memcpy(texbuffer, texbuffer1, sizeof(Color_float)* 256 * 256);

}

void ClearFrameBuffer()
{
	memset(screen_fb, 0xcfcfcf, 800 * 600 * 4);
// 	memcpy(zbuffer, zbufferMaxDepth, 800 * 600 * sizeof(float));
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
	result = (Color_float *)malloc(sizeof(Color_float));
	screen_init(800, 600, _T("Cube"));
	BindFB(800, 600, screen_fb);

	SetTimer(NULL,1, 1000, timerProc);
	SetCameraLookAt(&transformMatrix, eye_x, 0, 0);
	while (1/*screen_exit == 0 && screen_keys[VK_ESCAPE] == 0*/)
	{
// 		if (screen_keys[VK_SPACE])
// 			dir_light_or_point_light = -dir_light_or_point_light;
// 		if (_switch == 1)
// 			SetCameraLookAt(&transformMatrix, 3, 0, 0);
// 		else if (_switch == -1)
// 			SetCameraLookAt(&transformMatrix, -3, 0, 0);
// 		if (screen_keys[VK_LEFT])

		if (screen_keys[VK_UP])
		{
			eye_x -= 0.03f;
			SetCameraLookAt(&transformMatrix, eye_x, 0, 0);
		}
		if (screen_keys[VK_DOWN])
		{
			eye_x += 0.03f;
			SetCameraLookAt(&transformMatrix, eye_x, 0, 0);
		}
	{
			self_angle += 0.005f;
// 			self_angle -=90.0f;
			RotateCube(self_angle);
// 		if (_switch == 1)
// 			RotateCube((45.0f/180.0f) *3.141592654f);
// 		else if (_switch == -1)
// 			RotateCube(0);
		}

		DrawCube();

		screen_update();
// 		getchar();
		ClearFrameBuffer();
// 		Sleep(100);
	}
	return 0;
}