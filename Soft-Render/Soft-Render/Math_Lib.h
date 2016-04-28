#pragma once
#include <math.h>
#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  //=====================================================================
// 数学库：此部分应该不用详解，熟悉 D3D 矩阵变换即可
//=====================================================================
typedef struct { float m[4][4]; } matrix_t;
typedef struct { float x, y, z, w; } Pos;
typedef Pos point_t;

int CMID(int x, int min, int max) { return (x < min) ? min : ((x > max) ? max : x); }

// 计算插值：t 为 [0, 1] 之间的数值
float interp(float x1, float x2, float t) { return x1 + (x2 - x1) * t; }

float InvSqrt(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;         // get bits for floating value
	i = 0x5f375a86 - (i >> 1); // hidden initial guess
	x = *(float*)&i;          // convert bits back to float
	x = x*(1.5f - xhalf*x*x); // Newton step, repeating increases accuracy
							  //    x = x*(1.5f-xhalf*x*x); // add this in for added precision, or many more...
	return x;
} // InvSqrt

//| v |
float vector_length(const Pos *v) {
	float sq = v->x * v->x + v->y * v->y + v->z * v->z;
	return (float)sqrtf(sq);
}

// z = x + y
void vector_add(Pos *z, const Pos *x, const Pos *y) {
	z->x = x->x + y->x;
	z->y = x->y + y->y;
	z->z = x->z + y->z;
	z->w = 1.0f;
}

// z = x - y
void vector_sub(Pos *z, const Pos *x, const Pos *y) {
	z->x = x->x - y->x;
	z->y = x->y - y->y;
	z->z = x->z - y->z;
	z->w = 1.0f;
}
// 返回 x * y
Pos vector_multply(const Pos *x, const float y) {
	Pos z;
	z.x = x->x*y;
	z.y = x->y*y;
	z.z = x->z*y;
	z.w = x->w*y;
	return z;
}

// 矢量点乘
float vector_dotproduct(const Pos *x, const Pos *y) {

#if 0 //用SSE指令求，实际上这样存进去再求积再求和太慢了，还不如直接算
	float x_float[4] = { x->x,x->y,x->z, 0 }, y_float[4] = { y->x,y->y,y->z, 0 };
	__m128 mmx = _mm_load_ps(x_float);
	__m128 mmy = _mm_load_ps(y_float);
	__m128 lenSq = _mm_mul_ps(mmx, mmy);
//下面求平方的和
// xx = { xx3, xx2, xx1, xx0 }
	lenSq = _mm_hadd_ps(lenSq, lenSq);// xx = {xx3+xx2, xx1+xx0, xx3+xx2, xx1+xx0}
	lenSq = _mm_hadd_ps(lenSq, lenSq);// xx = {xx2+xx3+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0}
	float r; _mm_store_ss(&r, lenSq);
	return r;
#endif	
	return x->x * y->x + x->y * y->y + x->z * y->z;
}

// 矢量叉乘
void vector_crossproduct(Pos *z, const Pos *x, const Pos *y) {
	float m1, m2, m3;
	m1 = x->y * y->z - x->z * y->y;
	m2 = x->z * y->x - x->x * y->z;
	m3 = x->x * y->y - x->y * y->x;
	z->x = m1;
	z->y = m2;
	z->z = m3;
	z->w = 1.0f;
}

// 矢量插值，t取值 [0, 1]
void vector_interp(Pos *z, const Pos *x1, const Pos *x2, float t) {
	z->x = interp(x1->x, x2->x, t);
	z->y = interp(x1->y, x2->y, t);
	z->z = interp(x1->z, x2->z, t);
	z->w = 1.0f;
}

// 矢量归一化
void vector_normalize(Pos *v) {
//	float v_float[4] = { v->x,v->y,v->z, 0 }, fzero=0;
// 	__m128 mmv = _mm_load_ps(v_float);
// 	__m128 lenSq = _mm_mul_ps(mmv, mmv);
	//下面求平方的和
	// xx = { xx3, xx2, xx1, xx0 }
// 	lenSq = _mm_hadd_ps(lenSq, lenSq);// xx = {xx3+xx2, xx1+xx0, xx3+xx2, xx1+xx0}
// 	lenSq = _mm_hadd_ps(lenSq, lenSq);// xx = {xx2+xx3+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0, xx3+xx2+xx1+xx0}
// 	__m128 zero = _mm_load_ss(&fzero);
	const float lengthSq = v->x * v->x + v->y * v->y + v->z * v->z;
//	float length = vector_length(v);
// 	if (_mm_iszero_ss(lengthSq)) {
	if ((*(int*)&lengthSq) != 0) {
	
//	__m128i vcmp = _mm_castps_si128(_mm_cmpneq_ss(lenSq, zero)); // compare a, b for inequality
//	unsigned int test = _mm_movemask_epi8(vcmp); // extract results of comparison
	//if (test != 0xffff) {
//	if (_mm_cmpneq_ss(lenSq,zero) & 0xffffffff) {
		float inv;// = InvSqrt(lengthSq);
	//	float inv = 1.0f / length;
	__m128 in = _mm_load_ss(&lengthSq);
	//in = _mm_rsqrt_ss(lenSq);
	_mm_store_ss(&inv, _mm_rsqrt_ss(in));
		v->x *= inv;
		v->y *= inv;
		v->z *= inv;
	}
}

// c = a + b
void matrix_add(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] + b->m[i][j];
	}
}

// c = a - b
void matrix_sub(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] - b->m[i][j];
	}
}

// c = a * b
void matrix_mul(matrix_t *c, const matrix_t *a, const matrix_t *b) {
	matrix_t z;
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			z.m[j][i] = (a->m[j][0] * b->m[0][i]) +
				(a->m[j][1] * b->m[1][i]) +
				(a->m[j][2] * b->m[2][i]) +
				(a->m[j][3] * b->m[3][i]);
		}
	}
	c[0] = z;
}

// c = a * f
void matrix_scale(matrix_t *c, const matrix_t *a, float f) {
	int i, j;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
			c->m[i][j] = a->m[i][j] * f;
	}
}

// y = x * m
void matrix_apply(Pos *y, const Pos *x, const matrix_t *m) {
	float X = x->x, Y = x->y, Z = x->z, W = x->w;
	y->x = X * m->m[0][0] + Y * m->m[1][0] + Z * m->m[2][0] + W * m->m[3][0];
	y->y = X * m->m[0][1] + Y * m->m[1][1] + Z * m->m[2][1] + W * m->m[3][1];
	y->z = X * m->m[0][2] + Y * m->m[1][2] + Z * m->m[2][2] + W * m->m[3][2];
	y->w = X * m->m[0][3] + Y * m->m[1][3] + Z * m->m[2][3] + W * m->m[3][3];
}

void matrix_set_identity(matrix_t *m) {
	m->m[0][0] = m->m[1][1] = m->m[2][2] = m->m[3][3] = 1.0f;
	m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.0f;
	m->m[1][0] = m->m[1][2] = m->m[1][3] = 0.0f;
	m->m[2][0] = m->m[2][1] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.0f;
}

void matrix_set_zero(matrix_t *m) {
	m->m[0][0] = m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.0f;
	m->m[1][0] = m->m[1][1] = m->m[1][2] = m->m[1][3] = 0.0f;
	m->m[2][0] = m->m[2][1] = m->m[2][2] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = m->m[3][3] = 0.0f;
}

// 平移变换
void matrix_set_translate(matrix_t *m, float x, float y, float z) {
	matrix_set_identity(m);
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z;
}

// 缩放变换
void matrix_set_scale(matrix_t *m, float x, float y, float z) {
	matrix_set_identity(m);
	m->m[0][0] = x;
	m->m[1][1] = y;
	m->m[2][2] = z;
}

// 旋转矩阵
void matrix_set_rotate(matrix_t *m, float x, float y, float z, float theta) {
//只沿Z轴的旋转，调试用
// 	matrix_set_zero(m);
// 	m->m[0][0] = cos(theta);
// 	m->m[0][2] = sin(theta);
// 	m->m[1][0] = -sin(theta);
// 	m->m[1][1] = cos(theta);
// 	m->m[2][2] = 1.0f;
// 	return;
//调试用的结束
	float qsin = (float)sin(theta * 0.5f);
	float qcos = (float)cos(theta * 0.5f);
	Pos vec = { x, y, z, 1.0f };
	float w = qcos;
	vector_normalize(&vec);
	x = vec.x * qsin;
	y = vec.y * qsin;
	z = vec.z * qsin;
	m->m[0][0] = 1 - 2 * y * y - 2 * z * z;
	m->m[1][0] = 2 * x * y - 2 * w * z;
	m->m[2][0] = 2 * x * z + 2 * w * y;
	m->m[0][1] = 2 * x * y + 2 * w * z;
	m->m[1][1] = 1 - 2 * x * x - 2 * z * z;
	m->m[2][1] = 2 * y * z - 2 * w * x;
	m->m[0][2] = 2 * x * z - 2 * w * y;
	m->m[1][2] = 2 * y * z + 2 * w * x;
	m->m[2][2] = 1 - 2 * x * x - 2 * y * y;
	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.0f;
	m->m[3][3] = 1.0f;
}

// 设置摄像机
void matrix_set_lookat(matrix_t *m, const Pos *eye, const Pos *at, const Pos *up) {//来自 https://msdn.microsoft.com/zh-cn/library/windows/desktop/bb205342(v=vs.85).aspx
	Pos xaxis, yaxis, zaxis;

	vector_sub(&zaxis, at, eye);
	vector_normalize(&zaxis);
	vector_crossproduct(&xaxis, up, &zaxis);
	vector_normalize(&xaxis);
	vector_crossproduct(&yaxis, &zaxis, &xaxis);

	m->m[0][0] = xaxis.x;
	m->m[1][0] = xaxis.y;
	m->m[2][0] = xaxis.z;
	m->m[3][0] = -vector_dotproduct(&xaxis, eye);

	m->m[0][1] = yaxis.x;
	m->m[1][1] = yaxis.y;
	m->m[2][1] = yaxis.z;
	m->m[3][1] = -vector_dotproduct(&yaxis, eye);

	m->m[0][2] = zaxis.x;
	m->m[1][2] = zaxis.y;
	m->m[2][2] = zaxis.z;
	m->m[3][2] = -vector_dotproduct(&zaxis, eye);

	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.0f;
	m->m[3][3] = 1.0f;
}

// D3DXMatrixPerspectiveFovLH
void matrix_set_perspective(matrix_t *m, float fovy, float aspect, float zn, float zf) {//来自 https://msdn.microsoft.com/en-us/library/windows/desktop/bb205350(v=vs.85).aspx
	float fax = 1.0f / (float)tan(fovy * 0.5f);
	matrix_set_zero(m);
	m->m[0][0] = (float)(fax / aspect);
	m->m[1][1] = (float)(fax);
	m->m[2][2] = zf / (zf - zn);
	m->m[3][2] = -zn * zf / (zf - zn);
	m->m[2][3] = 1;
}