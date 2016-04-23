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
static float eye_x = 2, y = 0, z = 0;

unsigned int **framebufferPtr;      // 像素缓存：framebuffePtr[y] 代表第 y行
float **zbufferPtr;//深度缓冲
float *zbuffer;
float zbufferMaxDepth[800][600];//用来memcpy重置zbuffer的
Color_float *texbuffer;
Color_float texbuffer1[256][256];
Color_float **texbufferPtr;
float cube_camera_w[8];// 归一化前相机空间内顶点的w，暂存

typedef struct {
	matrix_t world;         // 世界坐标变换
	matrix_t view;          // 摄影机坐标变换
	matrix_t projection;    // 投影变换
	matrix_t transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}	TransformMatrix;

TransformMatrix transformMatrix;

void ApplyHomogenize()//透视变换到CVV，并归一化到NDC
{
	float rhw = 0;
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		rhw = 1.0f / cube_camera_w[i];
// 		cube_processed[i].rhw = rhw;
//		matrix_apply(& cube_processed[i].pos, &cube[i].pos, &transformMatrix.transform);//到相机空间
		cube_NDC_space[i].pos.x = (cube_camera_space[i].x * rhw + 1.0f) * 800.0f * 0.5f;
		cube_NDC_space[i].pos.y = (1.0f - cube_camera_space[i].y * rhw) * 600.0f * 0.5f;
		cube_NDC_space[i].pos.z = cube_camera_space[i].z * rhw;
		 cube_NDC_space[i].pos.w = cube_camera_w[i];
	};

}

void ApplyWVPTransform()//变换到cameraspace
{
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < 8; i++)
	{
		matrix_apply(&cube_camera_space[i], &cube[i].pos, &transformMatrix.transform);//到camera空间
	};
}
Color_float GetTexture(float u, float v)
{ 
// 	unsigned int nColorRef = RGB(0, 128, 255);//此时a为BGR
// 	int red = nColorRef & 255;//这样从COLORREF中获取RGB
// 	int green = nColorRef >> 8 & 255;
// 	int blue = nColorRef >> 16 & 255;
	if (u<0||v<0||u>255||v>255)
	{
		Color_float d = { 1, 0, 0 };
		return d;
	}
#if 1
// 	u *= 256.0f;
// 	v *= 256.0f;
	int x = u;
	int y = v;
	float u_ratio = u - x;
	float v_ratio = v - y;
	float u_opposite = 1 - u_ratio;
	float v_opposite = 1 - v_ratio;
// 	Color test = { 255, 126, 33 };
	 Color_float result;
	 result.r = ((texbufferPtr[x][y].r) * u_opposite + (texbufferPtr[x + 1][y].r) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].r) * u_opposite + (texbufferPtr[x + 1][y + 1].r) * u_ratio) * v_ratio;
	 result.g = ((texbufferPtr[x][y].g) * u_opposite + (texbufferPtr[x + 1][y].g) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].g) * u_opposite + (texbufferPtr[x + 1][y + 1].g) * u_ratio) * v_ratio;
	 result.b = ((texbufferPtr[x][y].b) * u_opposite + (texbufferPtr[x + 1][y].b) * u_ratio) * v_opposite + ((texbufferPtr[x][y + 1].b) * u_opposite + (texbufferPtr[x + 1][y + 1].b) * u_ratio) * v_ratio;
// 	int r = ((test.r) * u_opposite + (test.r) * u_ratio) * v_opposite + ((test.r) * u_opposite + (test.r) * u_ratio) * v_ratio;
// 	int g = ((test.g) * u_opposite + (test.g) * u_ratio) * v_opposite + ((test.g) * u_opposite + (test.g) * u_ratio) * v_ratio;
// 	int b = ((test.b) * u_opposite + (test.b) * u_ratio) * v_opposite + ((test.b) * u_opposite + (test.b) * u_ratio) * v_ratio;
// 	result = { r, g, b };
	return result;

#endif // 0
//  return texbuffer[(int)u][(int)v];
//	return a;

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
Pos dir_light_world = { 4, 0, 0, 1 };
void DrawPrimitive(Vertex *p1, Vertex* p2, Vertex* p3, Vector* triangleNormal_rotated_camera_normalized, Vector* p1_camera, Vector* p2_camera, Vector* p3_camera)//绘制图元
{
	if ((p2->pos.x - p1->pos.x) * (p3->pos.y - p1->pos.y) - (p3->pos.x - p1->pos.x) * (p2->pos.y - p1->pos.y) < 0.0f)// 背面消隐
		return;
	int i = 0, j = 0 ,x= 0,y=0;
#pragma region 定义
	float x1 = p1->pos.x;
	float y1 = p1->pos.y;
	float z1 = p1->pos.z;
	float x2 = p2->pos.x;
	float y2 = p2->pos.y;
	float z2 = p2->pos.z;
	float x3 = p3->pos.x;
	float y3 = p3->pos.y;
	float z3 = p3->pos.z;
	//下面是cameraspace里的
	float x1_camera = p1_camera->x;
	float y1_camera = p1_camera->y;
	float z1_camera = p1_camera->z;
	float x2_camera = p2_camera->x;
	float y2_camera = p2_camera->y;
	float z2_camera = p2_camera->z;
	float x3_camera = p3_camera->x;
	float y3_camera = p3_camera->y;
	float z3_camera = p3_camera->z;

#pragma endregion 
	float a, b, c;//方程系数
//	平面可以由Ax + By + Cz + D = 0来表示，因此z = f(x, y) = (-Ax - By - D) / C。
//	可以观察到，f(x + 1, y) – f(x, y) = -A / C，f(x, y + 1) – f(x, y) = -B / C，因此不必针对每个点都计算z值，只要有了三角形一个顶点的z值就可以用加减法算出其它点的z值。
	float A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
	float B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
	float C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
	float D = -(A*x1 + B*y1 + C*z1);
	float deltaZperX = -A / C;//以（x1,y1,z1）为起始点
	float deltaZperY = -B / C;
	//cameraspace中的三角形平面
// 	float A2 = (y2_camera - y1_camera) * (z3_camera - z1) - (z2_camera - z1_camera) * (y3_camera - y1_camera);
// 	float B2 = (z2_camera - z1_camera) * (x3_camera - x1) - (x2_camera - x1_camera) * (z3_camera - z1_camera);
// 	float C2 = (x2_camera - x1_camera) * (y3_camera - y1) - (y2_camera - y1_camera) * (x3_camera - x1_camera);
// 	float D2 = -(A*x1_camera + B*y1_camera + C*z1_camera);
// 	float deltaZperX2 = -A2 / C2;//以（x1,y1,z1）为起始点
// 	float deltaZperY2 = -B2 / C2;
// 
// 	float deltaXperY2 = -B2 / A2;//以（x1,y1,z1）为起始点
// 	float deltaXperZ2 = -C2 / A2;
// 
// 	float deltaYperX2 = -A2 / B2;//以（x1,y1,z1）为起始点
// 	float deltaYperZ2 = -C2 / B2;

	float z = 0;
	//纹理
	float u, v;

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

	//获得本三角形在cameraspace中所在的矩形
	float xmin_camera, xmax_camera, ymin_camera, ymax_camera;	
	xmin_camera = x1_camera < x2_camera ? x1_camera : x2_camera;
	xmin_camera = xmin_camera < x3_camera ? xmin_camera : x3_camera;
	xmax_camera = x1_camera > x2_camera ? x1_camera : x2_camera;
	xmax_camera = xmax_camera > x3_camera ? xmax : x3_camera;
	ymin_camera = y1_camera < y2_camera ? y1_camera : y2_camera;
	ymin_camera = ymin_camera < y3_camera ? ymin_camera : y3_camera;
	ymax_camera = y1_camera > y2_camera ? y1_camera : y2_camera;
	ymax_camera = ymax_camera > y3_camera ? ymax_camera : y3_camera;

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
	//逐像素光照
	Vector L;
	Pos dir_light_camera, dir_light_NDC;//向屏幕内照的方向光
// 	Pos t1,t2,t3,t4;//临时变量
// 	matrix_apply(&t3, &cube[0].pos, &transformMatrix.world);
// 	matrix_apply(&t1, &dir_light_world, &transformMatrix.view);
// 	matrix_apply(&t2, &t3, &transformMatrix.view);
// 	matrix_apply(&dir_light_camera, &t1, &transformMatrix.projection);
// 	matrix_apply(&t4, &t2, &transformMatrix.projection);
// 	float rhw = 1.0f / dir_light_camera.w;
// 	dir_light_NDC.x = (dir_light_camera.x * rhw + 1.0f) * 800.0f * 0.5f;
// 	dir_light_NDC.y = (1.0f - dir_light_camera.y * rhw) * 600.0f * 0.5f;
// 	dir_light_NDC.z = dir_light_camera.z * rhw;
// 	dir_light_NDC.w = dir_light_camera.w;

	Vector t1;
	matrix_apply(&t1, &dir_light_world, &transformMatrix.view);
	matrix_apply(&dir_light_camera, &t1, &transformMatrix.projection);

	float diffuse2;
// 	Pos p;// = { (op1->x + op2->x + op3->x + op4->x)*0.25f, (op1->y + op2->y + op3->y + op4->y)*0.25f, (op1->z + op2->z + op3->z + op4->z)*0.25f, 1 };
	Pos p_currentDrawing_camera;

// #pragma region Phong模型
// // 	Phong模型认为镜面反射的光强与反射光线和视线的夹角相关：
// // 		Ispec = Ks * Il * (dot(V, R)) ^ Ns
// // 		其中Ks 为镜面反射系数,Il是光源强度，Ns是高光指数，V表示从顶点到视点的观察方向，R代表反射光方向。由于反射光的方向R可以通过入射光方向L(从顶点指向光源)和物体的法向量求出，
// // 		R + L = 2 * dot(N, L) * N  即 R = 2 * dot(N, L) * N - L
// // 		所以最终的计算式为：
// // 		Ispec = Ks * Il * (dot(V, (2 * dot(N, L) * N – L)) ^ Ns
// 	float Ks = 0.7f, Il = 1.0f, Ns = 20.0f;
// 	Vector V; vector_sub(&V, &dir_light_world, &p_rotated_world);
// 	vector_normalize(&V);
// 	Vector t = vector_multply(&normal_rotated, 2.0f*vector_dotproduct(&L, &normal_rotated));
// 	Vector R; vector_sub(&R, &t, &L);
// 	vector_normalize(&R);
// 	float Ispec = Ks*Il*powf(vector_dotproduct(&V, &R), Ns);
// 	Ispec = max(Ispec, 0);
// // 		Ispec = min(Ispec, 1);
// #pragma endregion 

	//逐像素光照完
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
				z = z1 + (x - x1) *deltaZperX + (y - y1)*deltaZperY;//算该三角形的Z值
// 				if (z < (zbuffer[(int)y * 800 + (int)x]))//若离屏幕更近则显示否则舍弃
				{
					zr = a*(invp1w)+b*(invp2w)+c*(invp3w);
					invzr = 1.0f / zr;// a*(invp1w)+b*(invp2w)+c*(invp3w);
					u = ((a*(p1u *invp1w) + b*(p3u *invp2w) + c*(p3u *invp3w)) *invzr) * 254.0f;//w
					v = ((a*(p1v *invp1w) + b*(p2v *invp2w) + c*(p3v *invp3w)) *invzr) * 254.0f;//h
// 					zbuffer[(int)y * 800 + (int)x] = z;
					//算逐像素光照
					Pos temp = { x, y, z ,(p1w+p2w+p3w)};
					NDCToCameraSpace(&p_currentDrawing_camera, &temp);
					vector_sub(&L, &dir_light_camera, &p_currentDrawing_camera);
					vector_normalize(&L);
					diffuse2 = vector_dotproduct(&L, triangleNormal_rotated_camera_normalized);
					diffuse2 = max(diffuse2, 0);
// 					diffuse2 = min(diffuse2, 0.9f);
					//逐像素光照完
// 					const float inv255 = 1.0f / 255.0f;
					texColor = GetTexture(u, v);
					float r = texColor.r*diffuse2;
					float g = texColor.g*diffuse2;
					float b = texColor.b*diffuse2;

					framebufferPtr[y][x] = RGB(r*255.0f, g*255.0f, b*255.0f);
// 					framebufferPtr[x][y] = GetTexture(u, v);// *0x0000c0;
				}
			}
		}
	}
// 	getchar();
}

void DrawPlane(int a,int b,int c, int d)//绘制四边形，四个参数为顶点的索引
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
	//在这里就把光照算好算了...一个平面一个光照，对平行光来说
	Vector normal_camera = { 0,0,0,0 };
	Pos* op1 = &cube[a].pos, *op2 = &cube[b].pos, *op3 = &cube[c].pos, *op4 = &cube[d].pos;
	Pos* cp1 = &cube_camera_space[a], *cp2 = &cube_camera_space[b], *cp3 = &cube_camera_space[c], *cp4 = &cube_camera_space[d];

	// 	已知三点求三角形法向量(a,b,c)
// 		a = ((p2.y - p1.y)*(p3.z - p1.z) - (p2.z - p1.z)*(p3.y - p1.y));
// 		b = ((p2.z - p1.z)*(p3.x - p1.x) - (p2.x - p1.x)*(p3.z - p1.z));
// 		c = ((p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x));	
	// 	//这个算出的op面法向量是对的，按上面的公式法向量反了，不过这个是在worldspace下的，所以不用它
// 	normal.x = (op2->y - op3->y) * (op1->z - op3->z) - (op1->y - op3->y) * (op2->z - op3->z);
// 	normal.y = (op2->z - op3->z) * (op1->x - op3->x) - (op1->z - op3->z) * (op2->x - op3->x);
// 	normal.z = (op2->x - op3->x) * (op1->y - op3->y) - (op1->x - op3->x) * (op2->y - op3->y);
// 	//直接用p在cameraspace下的坐标
	normal_camera.x = (cp2->y - cp3->y) * (cp1->z - cp3->z) - (cp1->y - cp3->y) * (cp2->z - cp3->z);
	normal_camera.y = (cp2->z - cp3->z) * (cp1->x - cp3->x) - (cp1->z - cp3->z) * (cp2->x - cp3->x);
	normal_camera.z = (cp2->x - cp3->x) * (cp1->y - cp3->y) - (cp1->x - cp3->x) * (cp2->y - cp3->y);

// 	Vector L, normal_rotated, normal_rotated_camera;
// 	Vector dir_light_world = { 2,0,-0,0 };//向屏幕内照的方向光

// 	matrix_apply(&normal_rotated, &normal, &transformMatrix.world);
// 	matrix_apply(&normal_rotated_camera, &normal_rotated, &transformMatrix.projection);
// 	matrix_apply(&normal_rotated_camera, &normal, &transformMatrix.transform);
// 	vector_normalize(&normal_rotated);//光照模型里法线要规范化
// 	vector_normalize(&normal_rotated_camera);//光照模型里法线要规范化
// 	Vector p = { (op1->x + op2->x + op3->x + op4->x)*0.25f, (op1->y + op2->y + op3->y + op4->y)*0.25f, (op1->z + op2->z + op3->z + op4->z)*0.25f, 1 };
// 	matrix_apply(&p_rotated_world, &p, &transformMatrix.world);
// 	matrix_apply(&p_rotated_camera, &p_rotated_world, &transformMatrix.projection);
// 	matrix_apply(&p_rotated_camera, &p, &transformMatrix.transform);
// 	vector_sub(&L, &dir_light_world, &p_rotated_world);
// 	Vector R,temp;
// 	temp = vector_multply(normal, vector_dotproduct(&_2N, &L));
// 	vector_sub(&R, &temp, &L);
// 	float diffuse = vector_dotproduct(&L, &normal_rotated);
	
	//方向光算diffuse，都在cameraspace里算
	vector_normalize(&normal_camera);//光照模型里法线要规范化
// 	Vector p = { (cp1->x + cp2->x + cp3->x + cp4->x)*0.25f, (cp1->y + cp2->y + cp3->y + cp4->y)*0.25f, (cp1->z + cp2->z + cp3->z + cp4->z)*0.25f, 1 };
// 	Vector pNDC = { (p1->pos.x + p2->pos.x + p3->pos.x + p4->pos.x)*0.25f, (p1->pos.y + p2->pos.y + p3->pos.y + p4->pos.y)*0.25f, (p1->pos.z + p2->pos.z + p3->pos.z + p4->pos.z)*0.25f, 1 };
// 	Vector p_rotated_world; Vector p_rotated_camera; Vector dir_light_camera, dir_light_NDC;
// 	Vector t1;
// 	matrix_apply(&t1, &dir_light_world, &transformMatrix.view);
// 	matrix_apply(&dir_light_camera, &t1, &transformMatrix.projection);
// 	//把p1从NDC回到cameraspace，并放入test_p_in_cameraspace
// 	Pos test_p_in_cameraspace;
// 	NDCToCameraSpace(&test_p_in_cameraspace, &p1->pos);
// 	//放好了
// // 	Vector normal;
// // 	normal.x = (p2->pos.y - p3->pos.y) * (p1->pos.z - p3->pos.z) - (p1->pos.y - p3->pos.y) * (p2->pos.z - p3->pos.z);
// // 	normal.y = (p2->pos.z - p3->pos.z) * (p1->pos.x - p3->pos.x) - (p1->pos.z - p3->pos.z) * (p2->pos.x - p3->pos.x);
// // 	normal.z = (p2->pos.x - p3->pos.x) * (p1->pos.y - p3->pos.y) - (p1->pos.x - p3->pos.x) * (p2->pos.y - p3->pos.y);
// // 	normal.w = 0;
// // 	vector_normalize(&normal);
// // 	float rhw = 1.0f / dir_light_camera.w;
// // 	dir_light_NDC.x = (dir_light_camera.x * rhw + 1.0f) * 800.0f * 0.5f;
// // 	dir_light_NDC.y = (1.0f - dir_light_camera.y * rhw) * 600.0f * 0.5f;
// // 	dir_light_NDC.z = dir_light_camera.z * rhw;
// // 	dir_light_NDC.w = dir_light_camera.w;
// 	vector_sub(&L, &dir_light_camera, &test_p_in_cameraspace);
// 
// 	vector_normalize(&L);
// 	float diffuse = vector_dotproduct(&L, &normal_camera);
// 	diffuse = max(diffuse, 0);
// // 	diffuse = min(diffuse, 1);

// #pragma region Phong模型
// // 	Phong模型认为镜面反射的光强与反射光线和视线的夹角相关：
// // 		Ispec = Ks * Il * (dot(V, R)) ^ Ns
// // 		其中Ks 为镜面反射系数,Il是光源强度，Ns是高光指数，V表示从顶点到视点的观察方向，R代表反射光方向。由于反射光的方向R可以通过入射光方向L(从顶点指向光源)和物体的法向量求出，
// // 		R + L = 2 * dot(N, L) * N  即 R = 2 * dot(N, L) * N - L
// // 		所以最终的计算式为：
// // 		Ispec = Ks * Il * (dot(V, (2 * dot(N, L) * N – L)) ^ Ns
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
	//分成三角形图元绘制，逆时针为绘制方式
	DrawPrimitive(p3, p2, p1, &normal_camera, cp3, cp3, cp1);//传进去的cp是cameraspace下的顶点坐标
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
	ApplyWVPTransform();//顶点坐标从局部坐标系转到裁剪后的NDC
	for (size_t i = 0; i < 8; i++)
	{
		cube_camera_w[i] = cube_camera_space[i].w;
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

void RotateCube(float theta)//参数为弧度
{
	matrix_t m;
// 	matrix_set_rotate(&m, 0, 1, 0, theta);//沿y轴转
// 	matrix_set_rotate(&m, 1,0 , 0, theta);//沿x轴转
	matrix_set_rotate(&m, 0, 0, 1, theta);//沿z轴转
// 	matrix_set_rotate(&m, -1, -0.5, 1, theta);//算旋转矩阵
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
// 	framebufferPtr = (unsigned int **)ptr;//原来的
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
		framebuf = screen_fb;//framebuf指向fb的首地址

	for (j = 0; j < 600; j++)
	{
		framebufferPtr[j] = (unsigned int *)(framebuf + 800 * 4 * j);//framebuffer的每一行指向fb的每一行对应的地址
		zbufferPtr[j] = (float*)(zbuf + width * 4 * j);
	}
	for (j = 0; j < 256; j++)
	{
		texbufferPtr[j] = (Color_float *)(texbuf + 256 * sizeof(Color_float) * j);//framebuffer的每一行指向fb的每一行对应的地址
	}

	//贴图部分
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

	SetTimer(NULL,1, 1000, timerProc);
	SetCameraLookAt(&transformMatrix, eye_x, 0, 0);
	while (1/*screen_exit == 0 && screen_keys[VK_ESCAPE] == 0*/)
	{
		if (screen_keys[VK_SPACE])
			_switch = -_switch;
// 		if (_switch == 1)
// 			SetCameraLookAt(&transformMatrix, 3, 0, 0);
// 		else if (_switch == -1)
// 			SetCameraLookAt(&transformMatrix, -3, 0, 0);
// 		if (screen_keys[VK_LEFT])

		if (screen_keys[VK_UP])
		{
			eye_x -= 0.2;
			SetCameraLookAt(&transformMatrix, eye_x, 0, 0);
		}
		if (screen_keys[VK_DOWN])
		{
			eye_x += 0.2;
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