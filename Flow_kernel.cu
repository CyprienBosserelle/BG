//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////


// textures have to be declared here...
texture<float, 2, cudaReadModeElementType> texLZsBND;
texture<float, 2, cudaReadModeElementType> texRZsBND;
texture<float, 2, cudaReadModeElementType> texTZsBND;
texture<float, 2, cudaReadModeElementType> texBZsBND;

texture<float, 2, cudaReadModeElementType> texLUBND;
texture<float, 2, cudaReadModeElementType> texRUBND;
texture<float, 2, cudaReadModeElementType> texTUBND;
texture<float, 2, cudaReadModeElementType> texBUBND;

texture<float, 2, cudaReadModeElementType> texLVBND;
texture<float, 2, cudaReadModeElementType> texRVBND;
texture<float, 2, cudaReadModeElementType> texTVBND;
texture<float, 2, cudaReadModeElementType> texBVBND;

texture<float, 2, cudaReadModeElementType> texUWND;
texture<float, 2, cudaReadModeElementType> texVWND;
texture<float, 2, cudaReadModeElementType> texPATM;
texture<float, 2, cudaReadModeElementType> texRAIN;

template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};


template<class T>
__device__ T sq(T a)
{
	return a*a;
}

__device__ int findleftG(int ix,int iy,int leftblk, int ibl, int bdimx)
{
	int ileft;
	if (ix == 0)
	{
		if (leftblk != ibl)
		{
			ileft = 15 + iy * bdimx + leftblk * (bdimx*bdimx);
		}
		else
		{
			ileft = 0 + iy * bdimx + ibl*(bdimx*bdimx);
		}
	}
	else
	{
		ileft=(ix-1) + iy * bdimx + ibl*(bdimx*bdimx);
	}
	return ileft;
}

__device__ int findleftGSM(int ix, int iy, int leftblk, int ibl, int bdimx)
{
	int ileft;
	if (leftblk != ibl)
	{
			ileft = 15 + iy * bdimx + leftblk * (bdimx*bdimx);
	}
	else
	{
			ileft = 0 + iy * bdimx + ibl*(bdimx*bdimx);
	}
	
	return ileft;
}

__device__ int findrightG(int ix,int iy, int rightblk, int ibl, int bdimx)
{
	int iright;
	if (ix == (bdimx-1))
	{
		if (rightblk != ibl)
		{
			iright = 0 + iy * bdimx + rightblk * (bdimx*bdimx);
		}
		else
		{
			iright = 15 + iy * bdimx + ibl*(bdimx*bdimx);
		}
	}
	else
	{
		iright = (ix+1) + iy * bdimx + ibl*(bdimx*bdimx);
	}
	return iright;
}
__device__ int findrightGSM(int ix, int iy, int rightblk, int ibl, int bdimx)
{
	int iright;
	
		if (rightblk != ibl)
		{
			iright = 0 + iy * bdimx + rightblk * (bdimx*bdimx);
		}
		else
		{
			iright = 15 + iy * bdimx + ibl*(bdimx*bdimx);
		}
	
	return iright;
}

__device__ int findtopG(int ix,int iy, int topblk, int ibl, int bdimx)
{
	int itop;
	if (iy == (bdimx - 1))
	{
		if (topblk != ibl)// if it not refering to itself it has top neighbour
		{
			itop = ix + 0 * bdimx + topblk * (bdimx*bdimx);
		}
		else
		{
			itop = ix + 15 * bdimx + ibl*(bdimx*bdimx);
		}
	}
	else
	{
		itop = ix + (iy+1) * bdimx + ibl*(bdimx*bdimx);
	}
	return itop;
}
__device__ int findtopGSM(int ix, int iy, int topblk, int ibl, int bdimx)
{
	int itop;
	
		if (topblk != ibl)
		{
			itop = ix + 0 * bdimx + topblk * (bdimx*bdimx);
		}
		else
		{
			itop = ix + 15 * bdimx + ibl*(bdimx*bdimx);
		}
	
	return itop;
}

__device__ int findbotG(int ix,int iy, int botblk, int ibl, int bdimx)
{
	int ibot;
	if (iy == 0)
	{
		if (botblk != ibl)
		{
			ibot = ix + 15 * bdimx + botblk * (bdimx*bdimx);
		}
		else
		{
			ibot = ix + 0 * bdimx + ibl*(bdimx*bdimx);
		}
	}
	else
	{
		ibot = ix + (iy-1) * bdimx + ibl*(bdimx*bdimx);
	}
	return ibot;
}
__device__ int findbotGSM(int ix, int iy, int botblk, int ibl, int bdimx)
{
	int ibot;
	
		if (botblk != ibl)
		{
			ibot = ix + 15 * bdimx + botblk * (bdimx*bdimx);
		}
		else
		{
			ibot = ix + 0 * bdimx + ibl*(bdimx*bdimx);
		}
	
	return ibot;
}
__device__ float minmod2fGPU(float theta,float s0, float s1, float s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		float d1 = theta*(s1 - s0);
		float d2 = (s2 - s0) / 2.0f;
		float d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		float d1 = theta*(s1 - s0), d2 = (s2 - s0) / 2.0f, d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.;
}

template <class T> __device__ T minmod2GPU(T theta, T s0, T s1, T s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		T d1 = theta*(s1 - s0);
		T d2 = (s2 - s0) / T(2.0);
		T d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		T d1 = theta*(s1 - s0), d2 = (s2 - s0) / T(2.0), d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return T(0.0);
}

template <class T> __global__ void gradientGPUXY(int nx, int ny, T theta, T delta, T *a, T *dadx, T *dady)
{
	//
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;

	T a_i,a_r, a_l, a_t, a_b;

	//__shared__ float a_s[18][18];
	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		/*
		a_s[tx + 1][ty + 1] = a[ix + iy*nx];
		__syncthreads;
		// read the halo around the tile
		if (threadIdx.x == blockDim.x - 1)
			a_s[tx + 2][ty + 1] = a[xplus + iy*nx];

		if (threadIdx.x == 0)
			a_s[tx][ty + 1] = a[xminus + iy*nx];

		if (threadIdx.y == blockDim.y - 1)
			a_s[tx + 1][ty + 2] = a[ix + yplus*nx];

		if (threadIdx.y == 0)
			a_s[tx + 1][ty] = a[ix + yminus*nx];

		__syncthreads;
		*/
		a_i = a[ix + iy*nx];
		a_r= a[xplus + iy*nx];
		a_l= a[xminus + iy*nx];
		a_t = a[ix + yplus*nx];
		a_b = a[ix + yminus*nx];


		//dadx[i] = minmod2fGPU(theta, a_s[tx][ty + 1], a_s[tx + 1][ty + 1], a_s[tx + 2][ty + 1]) / delta;
		//dady[i] = minmod2fGPU(theta, a_s[tx + 1][ty], a_s[tx + 1][ty + 1], a_s[tx + 1][ty + 2]) / delta;

		dadx[i] = minmod2GPU(theta, a_l, a_i, a_r) / delta;
		dady[i] = minmod2GPU(theta, a_b, a_i, a_t) / delta;
	}

}


template <class T> __global__ void gradientGPUXYBUQ(T theta, T delta, int *leftblk, int *rightblk, int* topblk, int * botblk, T *a, T *dadx, T *dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int ix =  threadIdx.x;
	int iy = threadIdx.y;
	int ibl =  blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	
	
	
	int ileft, iright, itop, ibot;

	ileft = findleftG(ix,iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix,iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix,iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix,iy, botblk[ibl], ibl, blockDim.x);
	/*
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		printf("i= %i\t ileft=%i\t iright=%i\t itop=%i\t ibot=%i\n", i,ileft,iright,itop,ibot);
	}
	*/
	T a_i, a_r, a_l, a_t, a_b;
	/*
	__shared__ float a_s[18][18];
	
	

	
	a_s[ix][iy] = a[i];
	__syncthreads;
	//syncthread is needed here ?


	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	a_s[ix + 1][iy] = a[iright];

	if (threadIdx.x == 0)
	a_s[ix-1][iy] = a[ileft];

	if (threadIdx.y == blockDim.y - 1)
	a_s[ix ][iy + 1] = a[itop];

	if (threadIdx.y == 0)
	a_s[ix ][iy-1] = a[ibot];

	__syncthreads;
	*/
	a_i = a[i];
	a_r = a[iright];
	a_l = a[ileft];
	a_t = a[itop];
	a_b = a[ibot];
	
	/*
	dadx[i] = minmod2GPU(theta, a_s[ix-1][iy], a_s[ix][iy], a_s[ix + 1][iy]) / delta;
	dady[i] = minmod2GPU(theta, a_s[ix][iy-1], a_s[ix][iy], a_s[ix][iy + 1]) / delta;
	*/
	dadx[i] = minmod2GPU(theta, a_l, a_i, a_r) / delta;
	dady[i] = minmod2GPU(theta, a_b, a_i, a_t) / delta;
	

}

template <class T> __global__ void gradientGPUXYBUQSM(T theta, T delta, int *leftblk, int *rightblk, int* topblk, int * botblk, T *a, T *dadx, T *dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	// shared array index to make the code bit more readable
	int sx = ix + 1;
	int sy = iy + 1;



	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);



	
	int ileft, iright, itop, ibot;

	
	/*
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
	printf("i= %i\t ileft=%i\t iright=%i\t itop=%i\t ibot=%i\n", i,ileft,iright,itop,ibot);
	}
	*/
	//T a_i, a_r, a_l, a_t, a_b;
	
	__shared__ T a_s[18][18];




	a_s[sx][sy] = a[i];
	//__syncthreads;
	//syncthread is needed here ?


	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	{
		iright = findrightGSM(ix, iy, rightblk[ibl], ibl, blockDim.x);
		a_s[sx + 1][sy] = a[iright];
	}
	

	if (threadIdx.x == 0)
	{
		ileft = findleftGSM(ix, iy, leftblk[ibl], ibl, blockDim.x);
		a_s[sx-1][sy] = a[ileft];
	}
	

	if (threadIdx.y == blockDim.y - 1)
	{
		itop = findtopGSM(ix, iy, topblk[ibl], ibl, blockDim.x);
		a_s[sx][sy + 1] = a[itop];
	}

	if (threadIdx.y == 0)
	{
		ibot = findbotGSM(ix, iy, botblk[ibl], ibl, blockDim.x);
		a_s[sx][sy - 1] = a[ibot];
	}

	__syncthreads;
	/*
	a_i = a[i];
	a_r = a[iright];
	a_l = a[ileft];
	a_t = a[itop];
	a_b = a[ibot];
	*/
	
	dadx[i] = minmod2GPU(theta, a_s[sx-1][sy], a_s[sx][sy], a_s[sx + 1][sy]) / delta;
	dady[i] = minmod2GPU(theta, a_s[sx][sy-1], a_s[sx][sy], a_s[sx][sy + 1]) / delta;
	/*
	dadx[i] = minmod2GPU(theta, a_l, a_i, a_r) / delta;
	dady[i] = minmod2GPU(theta, a_b, a_i, a_t) / delta;
	*/

}

template <class T>
__global__ void interp2ATMP(float xoatm,float yoatm,float dxatm,T delta, T Pref,T*blockxo, T *blockyo,  T * P)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	T x = blockxo[ibl] + ix*delta;
	T y = blockyo[ibl] + iy*delta;

	float Pw;

	Pw = tex2D(texPATM, (x - xoatm) / dxatm + 0.5, (y - yoatm) / dxatm + 0.5);

	P[i] = Pw - Pref;
	

}
__global__ void gradientGPUX(int nx, int ny,float theta, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = ix + iy*nx;
	int xplus, xminus;

	__shared__ float a_s[16][16]; // Hard wired stuff Be carefull
	__shared__ float al_s[16][16];
	__shared__ float ar_s[16][16];
	//float dadxi;
	if (ix < nx && iy < ny)
	{
		//
		//
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);

		i = ix + iy*nx;
		//a_s[tx][ty] = a[ix + iy*nx];
		//al_s[tx][ty] = a[xminus + iy*nx];
		//ar_s[tx][ty] = a[xplus + iy*nx];
		
		a_s[tx][ty] = a[ix + iy*nx];

		// read the halo around the tile
		__syncthreads;

		al_s[tx][ty] = a[xminus + iy*nx];
		__syncthreads;
		ar_s[tx][ty] = a[xplus + iy*nx];
		
		// Need to wait for threadX 0 and threadX 16-1 to finish
		__syncthreads;
		




		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		//dadx[i] = minmod2fGPU(theta,a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;
		// These below are somewhat slower when using shared mem. I'm unsure why (bank conflict?)
		//dadx[i] = minmod2fGPU(theta, al_s[tx][ty], a_s[tx][ty], ar_s[tx][ty]) / delta;
		//dadx[i] = minmod2fGPU(theta, a_s[tx][ty], a_s[tx+1][ty], a_s[tx+2][ty]) / delta;
		//__device__ float minmod2fGPU(float theta,float s0, float s1, float s2)


		//float d1, d2, d3;
		//float s0, s1, s2;
		
		//dadxi = 0.0f;
		dadx[i] = minmod2fGPU(theta, al_s[tx][ty], a_s[tx][ty], ar_s[tx][ty]) / delta;
		/*
		s0 = al_s[tx][ty];// there will be bank conflict here
		s1 = a_s[tx][ty];// there will be bank conflict here
		s2 = ar_s[tx][ty];// there will be bank conflict here

		if (s0 < s1 && s1 < s2) {
			d1 = theta*(s1 - s0);
			d2 = (s2 - s0) / 2.0f;
			d3 = theta*(s2 - s1);
			if (d2 < d1) d1 = d2;
			dadxi = min(d1, d3);
		}
		if (s0 > s1 && s1 > s2) {
			d1 = theta*(s1 - s0);
			d2 = (s2 - s0) / 2.0f;
			d3 = theta*(s2 - s1);
			if (d2 > d1) d1 = d2;
			dadxi = max(d1, d3);
		}
		
		dadx[i] = dadxi / delta;
		*/
		
	}



}

__global__ void gradientGPUXOLD(int nx, int ny, float theta, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, xminus;

	if (ix < nx && iy < ny)
	{
		//
		//

		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dadx[i] = minmod2fGPU(theta, a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;

	}



}
__global__ void gradientGPUYSM(int nx, int ny, float theta, float delta, float *a, float *dady)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = ix + iy*nx;
	int  yplus, yminus;

	__shared__ float a_s[16][18];

	float dadyi = 0.0f;
	if (ix < nx && iy < ny)
	{
		//
		//

		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;

		a_s[tx][ty] = a[ix + yminus*nx];

		// read the halo around the tile

		if (threadIdx.x == 15)//blockDim.x - 1
		{
			a_s[tx][ty + 1] = a[i];
			a_s[tx][ty + 2] = a[ix + yplus*nx];

		}

		// Need to wait for threadX 0 and threadX 16-1 to finish
		__syncthreads;

		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		//dady[i] = minmod2fGPU(theta, a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;
		//dady[i] = minmod2fGPU(theta, a_s[tx][ty], a_s[tx][ty + 1], a_s[tx][ty + 2]) / delta;
		float d1, d2, d3;
		if (a_s[tx][ty] < a_s[tx][ty + 1] && a_s[tx][ty + 1] < a_s[tx][ty + 2]) {
			d1 = theta*(a_s[tx][ty + 1] - a_s[tx][ty]);
			d2 = (a_s[tx][ty + 2] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx][ty + 2] - a_s[tx][ty + 1]);
			if (d2 < d1) d1 = d2;
			dadyi = min(d1, d3);
		}
		if (a_s[tx][ty] > a_s[tx][ty + 1] && a_s[tx][ty + 1] > a_s[tx][ty + 2]) {
			d1 = theta*(a_s[tx][ty + 1] - a_s[tx][ty]);
			d2 = (a_s[tx][ty + 2] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx][ty + 2] - a_s[tx][ty + 1]);
			if (d2 > d1) d1 = d2;
			dadyi = max(d1, d3);
		}
		dady[i] = dadyi / delta;
	}



}
__global__ void gradientGPUY(int nx, int ny,float theta, float delta, float *a, float *dady)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  yplus, yminus;

	if (ix < nx && iy < ny)
	{
		//
		//
		
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dady[i] = minmod2fGPU(theta,a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;

	}



}


__global__ void updateKurgX( float delta, float g, float eps,float CFL, int *leftblk, float * hh, float *zs, float *uu, float * vv, float *dzsdx, float *dhdx, float * dudx, float *dvdx, float *Fhu, float *Fqux, float *Fqvx, float *Su, float * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	//float epsc = 0.07f;

	// This is based on kurganov and Petrova 2007


	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		



		float dhdxi= dhdx[i];
		float dhdxmin = dhdx[ileft];
		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		float hi = hh[i];

		float hn = hh[ileft];


		if (hi > eps || hn > eps)
		{
			float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm,sl,sr;

			// along X
			dx = delta*0.5f;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[ileft] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[ileft] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0f, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[ileft] + dx*dudx[ileft];
			hm = max(0.0f, hr + zr - zlr);

			float fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga,apm;
			float epsi = 1e-30f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(max(up + cp, um + cmo),0.0f);// eq. 2.22
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo),0.0f);//eq. 2.23
			//am = min(am, 0.0f);

			ad = 1.0f / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um; 
			qp = hp*up;
			//qm = hm*um*(sqrtf(2) / sqrtf(1.0f + max(1.0f, powf(epsc / hm,4.0f))));
			//qp = hp*up*(sqrtf(2) / sqrtf(1.0f + max(1.0f, powf(epsc / hp,4.0f))));

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5f;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad; // H  in eq. 2.24 or eq 3.7 for F(h)
				fu = (ap*(qm*um + ga*hm2 ) - am*(qp*up + ga*hp2 ) + apm*(qp - qm)) *ad; // Eq 3.7 second term (X direction)
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				

			}
			else
			{
				fh = 0.0f;
				fu = 0.0f;
				dtmax[i] = 1.0f / 1e-30f;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/


			
			if (fh > 0.0f)
			{
				fv = (vv[ileft] + dx*dvdx[ileft])*fh;// Eq 3.7 third term? (X direction)
			}
			else 
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
				printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0f / 1e-30f;
			Fhu[i] = 0.0f;
			Fqux[i] = 0.0f;
			Su[i] = 0.0f;
			Fqvx[i] = 0.0f;
		}

	}


}

template <class T>
__global__ void updateKurgXATM(T delta, T g, T eps, T CFL, T Pa2m, int *leftblk, T * hh, T *zs, T *uu, T * vv, T *Patm, T *dzsdx, T *dhdx, T * dudx, T *dvdx, T *dpdx, T *Fhu, T *Fqux, T *Fqvx, T *Su, T * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);



		


		T dhdxi = dhdx[i];
		T dhdxmin = dhdx[ileft];
		T cm = (T)1.0;// 0.1;
		T fmu = (T)1.0;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		T hi = hh[i];

		T hn = hh[ileft];


		if (hi > eps || hn > eps)
		{
			T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;

			// along X
			dx = delta*T(0.5);
			zi = zs[i] - hi + Pa2m * Patm[i];

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi + Pa2m*dpdx[i]);
			//printf("%f\n", zl);

			zn = zs[ileft] - hn + Pa2m * Patm[ileft];

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[ileft] - dhdxmin + Pa2m*dpdx[ileft]);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max((T)0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[ileft] + dx*dudx[ileft];
			hm = max((T)0.0, hr + zr - zlr);

			T fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			T cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			T epsi = (T)1e-30;

			//T epsc = T(0.07);

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(max(up + cp, um + cmo), (T)0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), (T)0.0);
			//am = min(am, 0.0f);
			ad = T(1.0) / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrt(T(2.)) / sqrt(T(1.0) + max(T(1.0), (T)pow((T)epsc / hm, (T)4.0))));
			//qp = hp*up*(sqrt(T(2.)) / sqrt(T(1.0) + max(T(1.0), (T)pow((T)epsc / hp, (T)4.0))));

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*T(0.5);
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				T dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				

			}
			else
			{
				fh = (T)0.0;
				fu = (T)0.0;
				dtmax[i] = (T)1.0 / (T)1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > (T)0.0)
			{
				fv = (vv[ileft] + dx*dvdx[ileft])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = (T)1.0 / (T) 1e-30;
			Fhu[i] = (T) 0.0;
			Fqux[i] = (T) 0.0;
			Su[i] = (T) 0.0;
			Fqvx[i] = (T) 0.0;
		}

	}


}


__global__ void updateKurgXD( double delta, double g, double eps, double CFL, int *leftblk, double * hh, double *zs, double *uu, double * vv, double *dzsdx, double *dhdx, double * dudx, double *dvdx, double *Fhu, double *Fqux, double *Fqvx, double *Su, double * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);






		double dhdxi = dhdx[i];
		double dhdxmin = dhdx[ileft];
		double cm = 1.0;// 0.1;
		double fmu = 1.0;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		double hi = hh[i];

		double hn = hh[ileft];


		if (hi > eps || hn > eps)
		{
			double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;


			double epsc = pow(delta,4.0);
			// along X
			dx = delta*0.5;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[ileft] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[ileft] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[ileft] + dx*dudx[ileft];
			hm = max(0.0, hr + zr - zlr);

			double fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			//Correct for spurious currents in really shallow depth
			//qm = hm*um;
			//qp = hp*up;
			qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));
			qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				

			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0)
			{
				fv = (vv[ileft] + dx*dvdx[ileft])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhu[i] = 0.0;
			Fqux[i] = 0.0;
			Su[i] = 0.0;
			Fqvx[i] = 0.0;
		}

	}


}

__global__ void updateKurgXSPH( double delta, double g, double eps, double CFL,int *leftblk, double *blockyo, double Radius, double * hh, double *zs, double *uu, double * vv, double *dzsdx, double *dhdx, double * dudx, double *dvdx, double *Fhu, double *Fqux, double *Fqvx, double *Su, double * dtmax)
{
	//Same as updateKurgX but with Spherical coordinates
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	double cm, fmu,y,phi,dphi;

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);






		double dhdxi = dhdx[i];
		double dhdxmin = dhdx[ileft];

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		//fmv = cosf(phi);

		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		double hi = hh[i];

		double hn = hh[ileft];


		if (hi > eps || hn > eps)
		{
			double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;

			//double epsc = 0.07;
			// along X
			dx = delta*0.5;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[ileft] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[ileft] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[ileft] + dx*dudx[ileft];
			hm = max(0.0, hr + zr - zlr);

			double fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am,ad;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));
			//qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				
			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0)
			{
				fv = (vv[ileft] + dx*dvdx[ileft])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhu[i] = 0.0;
			Fqux[i] = 0.0;
			Su[i] = 0.0;
			Fqvx[i] = 0.0;
		}

	}


}

__global__ void updateKurgXSPHATM(double delta, double g, double eps, double CFL, double Pa2m, int *leftblk, double *blockyo, double Radius, double * hh, double *zs, double *uu, double * vv, double *Patm, double *dzsdx, double *dhdx, double * dudx, double *dvdx, double *dpdx, double *Fhu, double *Fqux, double *Fqvx, double *Su, double * dtmax)
{
	//Same as updateKurgX but with Spherical coordinates
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	double cm, fmu, y, phi, dphi;

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);






		double dhdxi = dhdx[i];
		double dhdxmin = dhdx[ileft];

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		//fmv = cosf(phi);

		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		double hi = hh[i];

		double hn = hh[ileft];


		if (hi > eps || hn > eps)
		{
			double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;

			//double epsc = 0.07;
			// along X
			dx = delta*0.5;
			zi = zs[i] - hi + Pa2m * Patm[i];

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi + Pa2m*dpdx[i]);
			//printf("%f\n", zl);

			zn = zs[ileft] - hn + Pa2m * Patm[ileft];

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[ileft] - dhdxmin + Pa2m*dpdx[ileft]);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[ileft] + dx*dudx[ileft];
			hm = max(0.0, hr + zr - zlr);

			double fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am, ad;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));
			//qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;
				

			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0)
			{
				fv = (vv[ileft] + dx*dvdx[ileft])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhu[i] = 0.0;
			Fqux[i] = 0.0;
			Su[i] = 0.0;
			Fqvx[i] = 0.0;
		}

	}


}

__global__ void updateKurgY(float delta, float g, float eps, float CFL, int *botblk, float * hh, float *zs, float *uu, float * vv, float *dzsdy, float *dhdy, float * dudy, float *dvdy, float *Fhv, float *Fqvy, float *Fquy, float *Sv, float * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		float cm = 1.0f;// 0.1;
		//float fmu = 1.0;
		float fmv = 1.0f;

		//__shared__ float hi[16][16];
		float dhdyi = dhdy[i];
		float dhdymin = dhdy[ibot];
		float hi = hh[i];
		float hn = hh[ibot];
		float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ibot];
			dx = delta / 2.0f;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ibot] - hn;
			zr = zn + dx*(dzsdy[ibot] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.f, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ibot] + dx*dvdy[ibot];
			hm = max(0.f, hr + zr - zlr);

			//// Reimann solver
			float fh, fu, fv,sl,sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt,ad,hm2,hp2,ga,apm;
			float epsi = 1e-30f;
			//float epsc = 0.07f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(max(up + cp, um + cmo),0.0f);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo),0.0f);
			//am = min(am, 0.0f);
			ad = 1.0f / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrtf(2.0f) / sqrtf(1.0f + max(1.0f, powf(epsc / hm, 4.0f))));
			//qp = hp*up*(sqrtf(2.0f) / sqrtf(1.0f + max(1.0f, powf(epsc / hp, 4.0f))));

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5f;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2 ) - am*(qp*up + ga*hp2) +	apm*(qp - qm)) *ad;
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;
				
				

			}
			else
			{
				fh = 0.0f;
				fu = 0.0f;
				dtmax[i] = 1.0f / 1e-30f;
			}

			if (fh > 0.0f)
			{
				fv = (uu[ibot] + dx*dudy[ibot])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0f / 1e-30f;
			Fhv[i] = 0.0f;
			Fqvy[i] = 0.0f;
			Sv[i] = 0.0f;
			Fquy[i] = 0.0f;
		}
	}
}

template <class T>
__global__ void updateKurgYATM(T delta, T g, T eps, T CFL, T Pa2m, int *botblk, T * hh, T *zs, T *uu, T * vv, T *Patm, T *dzsdy, T *dhdy, T * dudy, T *dvdy,T *dpdy, T *Fhv, T *Fqvy, T *Fquy, T *Sv, T * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		T cm = (T) 1.0;// 0.1;
						//float fmu = 1.0;
		T fmv = (T) 1.0f;

		//__shared__ float hi[16][16];
		T dhdyi = dhdy[i];
		T dhdymin = dhdy[ibot];
		T hi = hh[i];
		T hn = hh[ibot];
		T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ibot];
			dx = delta / ((T) 2.0);
			zi = zs[i] - hi + Pa2m * Patm[i];
			zl = zi - dx*(dzsdy[i] - dhdyi+Pa2m *dpdy[i]);
			zn = zs[ibot] - hn + Pa2m * Patm[ibot];
			zr = zn + dx*(dzsdy[ibot] - dhdymin + Pa2m *dpdy[ibot]);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max((T) 0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ibot] + dx*dvdy[ibot];
			hm = max((T) 0.0, hr + zr - zlr);

			//// Reimann solver
			T fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			T cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			T epsi = (T)1e-30;

			//T epsc = (T)0.07;

			cp = sqrt(g*hp);/// how to enforce sqrtf when T is float ?
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), (T) 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), (T) 0.0);
			//am = min(am, 0.0f);
			ad = 1.0f / (ap - am);
			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrt(T(2.0)) / sqrt(T(1.0) + max(T(1.0), (T)pow((T)epsc / hm, (T)4.0))));
			//qp = hp*up*(sqrt(T(2.0)) / sqrt(T(1.0) + max(T(1.0), (T)pow((T)epsc / hp, (T)4.0))));

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*T(0.5);
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;
				
				

			}
			else
			{
				fh = (T) 0.0;
				fu = (T) 0.0;
				dtmax[i] = (T) 1.0 / ((T) 1e-30);
			}

			if (fh > (T)0.0)
			{
				fv = (uu[ibot] + dx*dudy[ibot])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = (T)1.0 / ((T) 1e-30);
			Fhv[i] = (T)0.0;
			Fqvy[i] = (T) 0.0;
			Sv[i] = (T)0.0;
			Fquy[i] = (T)0.0;
		}
	}
}

__global__ void updateKurgYD( double delta, double g, double eps, double CFL, int *botblk, double * hh, double *zs, double *uu, double * vv, double *dzsdy, double *dhdy, double * dudy, double *dvdy, double *Fhv, double *Fqvy, double *Fquy, double *Sv, double * dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		double cm = 1.0;// 0.1;
						//float fmu = 1.0;
		double fmv = 1.0;

		//__shared__ float hi[16][16];
		double dhdyi = dhdy[i];
		double dhdymin = dhdy[ibot];
		double hi = hh[i];
		double hn = hh[ibot];
		double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

		double epsc = pow(delta, 4.0);


		if (hi > eps || hn > eps)
		{
			hn = hh[ibot];
			dx = delta / 2.0;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ibot] - hn;
			zr = zn + dx*(dzsdy[ibot] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ibot] + dx*dvdy[ibot];
			hm = max(0.0, hr + zr - zlr);

			//// Reimann solver
			double fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			double epsi = 1e-30;
			//double epsc = 0.07;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			//Correct for spurious currents in really shallow depth
			//qm = hm*um;
			//qp = hp*up;
			qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));
			qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;
				

			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}

			if (fh > 0.0f)
			{
				fv = (uu[ibot] + dx*dudy[ibot])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhv[i] = 0.0;
			Fqvy[i] = 0.0;
			Sv[i] = 0.0;
			Fquy[i] = 0.0;
		}
	}
}
__global__ void updateKurgYSPH( double delta, double g, double eps, double CFL,int *botblk, double * blockyo, double Radius, double * hh, double *zs, double *uu, double * vv, double *dzsdy, double *dhdy, double * dudy, double *dvdy, double *Fhv, double *Fqvy, double *Fquy, double *Sv, double * dtmax)
{
	// Same as updateKurgY but with Spherical coordinate corrections
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);


	double cm, fmv, phi, dphi, y;

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		//fmu = 1.0f;
		fmv = cos(phi);

		//float cm = 1.0f;// 0.1;
						//float fmu = 1.0;
		//float fmv = 1.0f;

		//__shared__ float hi[16][16];
		double dhdyi = dhdy[i];
		double dhdymin = dhdy[ibot];
		double hi = hh[i];
		double hn = hh[ibot];
		double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ibot];
			dx = delta / 2.;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ibot] - hn;
			zr = zn + dx*(dzsdy[ibot] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ibot] + dx*dvdy[ibot];
			hm = max(0.0, hr + zr - zlr);

			//// Reimann solver
			double fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am, ad;
			double epsi = 1e-30;

			//double epsc = 0.07;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);

			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up;
			//qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));
			//qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				
			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}

			if (fh > 0.0)
			{
				fv = (uu[ibot] + dx*dudy[ibot])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhv[i] = 0.0;
			Fqvy[i] = 0.0;
			Sv[i] = 0.0;
			Fquy[i] = 0.0;
		}
	}
}

__global__ void updateKurgYSPHATM(double delta, double g, double eps, double CFL, double Pa2m, int *botblk, double * blockyo, double Radius, double * hh, double *zs, double *uu, double * vv, double *Patm, double *dzsdy, double *dhdy, double * dudy, double *dvdy, double *dpdy, double *Fhv, double *Fqvy, double *Fquy, double *Sv, double * dtmax)
{
	// Same as updateKurgY but with Spherical coordinate corrections
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);


	double cm, fmv, phi, dphi, y;

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		//fmu = 1.0f;
		fmv = cos(phi);

		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0;
		//float fmv = 1.0f;

		//__shared__ float hi[16][16];
		double dhdyi = dhdy[i];
		double dhdymin = dhdy[ibot];
		double hi = hh[i];
		double hn = hh[ibot];
		double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ibot];
			dx = delta / 2.;
			zi = zs[i] - hi + Pa2m * Patm[i];
			zl = zi - dx*(dzsdy[i] - dhdyi + Pa2m *dpdy[i]);
			zn = zs[ibot] - hn + Pa2m * Patm[ibot];
			zr = zn + dx*(dzsdy[ibot] - dhdymin + Pa2m *dpdy[ibot]);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ibot] + dx*dvdy[ibot];
			hm = max(0.0, hr + zr - zlr);

			//// Reimann solver
			double fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am, ad;
			double epsi = 1e-30;

			//double epsc = 0.07;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);

			//Correct for spurious currents in really shallow depth
			qm = hm*um;
			qp = hp*up; 
			//qm = hm*um*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hm, 4.0))));;
			//qp = hp*up*(sqrt(2.0) / sqrt(1.0 + max(1.0, pow(epsc / hp, 4.0))));;

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;

				
			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}

			if (fh > 0.0)
			{
				fv = (uu[ibot] + dx*dudy[ibot])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhv[i] = 0.0;
			Fqvy[i] = 0.0;
			Sv[i] = 0.0;
			Fquy[i] = 0.0;
		}
	}
}

template<class T>
__global__ void uvcorr(T delta, T* hh, T*uu, T*vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	// Corect u for spurious velocities t very shallow depth
	//u=sqrt(2)*h*hu/(sqrt(h^4+max(h^4,epsdepth)))
	T epsdelta = delta*delta*delta*delta;

	T uui,hhi,vvi, hhiQ;

	uui = uu[i];
	vvi = vv[i];
	hhi = hh[i];

	hhiQ = hhi*hhi*hhi*hhi;



	uu[i] = sqrt(T(2.0))*hhi*(hhi*uui) / (sqrt(hhiQ + max(epsdelta, hhiQ)));
	vv[i] = sqrt(T(2.0))*hhi*(hhi*vvi) / (sqrt(hhiQ + max(epsdelta, hhiQ)));






}

__global__ void updateEV( float delta, float g, float fc, int *rightblk, int*topblk, float * hh, float *uu, float * vv, float * Fhu, float *Fhv, float * Su, float *Sv, float *Fqux, float *Fquy, float *Fqvx, float *Fqvy, float *dh, float *dhu, float *dhv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright,itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		float fmv = 1.0f;

		float hi = hh[i];
		float uui = uu[i];
		float vvi = vv[i];


		float cmdinv, ga;

		cmdinv = 1.0f / (cm*delta);
		ga = 0.5f*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0f*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
		float dmdt = (fmv - fmv) / (cm*delta);
		float fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + fc*hi*vvi;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv - fc*hi*uui;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);// This term is == 0 so should be commented here
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);// Need double checking before doing that
	}
}


template <class T>
__global__ void updateEVATM(T delta, T g, T fc, T xowind, T yowind ,T dxwind, T Cd, int *rightblk, int*topblk, T * blockxo, T* blockyo, T * hh, T *uu, T * vv, T * Fhu, T *Fhv, T * Su, T *Sv, T *Fqux, T *Fquy, T *Fqvx, T *Fqvy, T *dh, T *dhu, T *dhv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	T Uw = 0.0;
	T Vw = 0.0;


	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	T x = blockxo[ibl] + ix*delta;
	T y = blockyo[ibl] + iy*delta;


	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	Uw=(T) tex2D(texUWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5); //tex2d return a float always!
	Vw = (T) tex2D(texVWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);
	
	//xplus = min(ix + 1, nx - 1);
	//xminus = max(ix - 1, 0);
	//yplus = min(iy + 1, ny - 1);
	//yminus = max(iy - 1, 0);

	T cm = (T) 1.0;// 0.1;
	T fmu = (T) 1.0;
	T fmv = (T) 1.0;

	T hi = hh[i];
	T uui = uu[i];
	T vvi = vv[i];


	T cmdinv, ga;

	cmdinv = (T) 1.0 / (cm*delta);
	ga = (T) 0.5*g;
	////
	//vector dhu = vector(updates[1 + dimension*l]);
	//foreach() {
	//	double dhl =
	//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
	//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
	//	foreach_dimension()
	//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
	//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
	//float cm = 1.0;

	dh[i] = T(-1.0)*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
	//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


	//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
	//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
	T dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
	T dmdt = (fmv - fmv) / (cm*delta);
	T fG = vvi * dmdl - uui * dmdt;
	dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + T(0.00121951)*Cd*Uw*abs(Uw) + fc*hi*vvi;
	dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv + T(0.00121951)*Cd*Vw*abs(Vw) - fc*hi*uui;
	//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
	dhu[i] += hi * (ga*hi *dmdl + fG*vvi);// This term is == 0 so should be commented here
	dhv[i] += hi * (ga*hi *dmdt - fG*uui);// Need double checking before doing that
	
}


template <class T>
__global__ void updateEVATMWUNI(T delta, T g, T fc, T uwind, T vwind, T Cd, int *rightblk, int*topblk, T * hh, T *uu, T * vv, T * Fhu, T *Fhv, T * Su, T *Sv, T *Fqux, T *Fquy, T *Fqvx, T *Fqvy, T *dh, T *dhu, T *dhv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	T Uw = T(0.0);
	T Vw = T(0.0);


	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	//float x = blockxo[ibl] + ix*delta;
	//float y = blockyo[ibl] + iy*delta;


	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	Uw = uwind;// tex2D(texUWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);
	Vw = vwind;// tex2D(texVWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);

	//xplus = min(ix + 1, nx - 1);
	//xminus = max(ix - 1, 0);
	//yplus = min(iy + 1, ny - 1);
	//yminus = max(iy - 1, 0);

	T cm = T(1.0);// 0.1;
	T fmu = T(1.0);
	T fmv = T(1.0);

	T hi = hh[i];
	T uui = uu[i];
	T vvi = vv[i];


	T cmdinv, ga;

	cmdinv = T(1.0) / (cm*delta);
	ga = T(0.5)*g;
	////
	//vector dhu = vector(updates[1 + dimension*l]);
	//foreach() {
	//	double dhl =
	//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
	//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
	//	foreach_dimension()
	//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
	//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
	//float cm = 1.0;

	dh[i] = T(-1.0)*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
	//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


	//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
	//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
	T dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
	T dmdt = (fmv - fmv) / (cm*delta);
	T fG = vvi * dmdl - uui * dmdt;
	dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + T(0.00121951)*Cd*Uw*abs(Uw) + fc*hi*vvi;
	dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv + T(0.00121951)*Cd*Vw*abs(Vw) - fc*hi*uui;
	//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
	dhu[i] += hi * (ga*hi *dmdl + fG*vvi);// This term is == 0 so should be commented here
	dhv[i] += hi * (ga*hi *dmdt - fG*uui);// Need double checking before doing that

}

__global__ void updateEVD( double delta, double g, double fc, int *rightblk, int*topblk, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		double cm = 1.0;// 0.1;
		double fmu = 1.0;
		double fmv = 1.0;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		double cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		double dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
		double dmdt = (fmv - fmv) / (cm*delta);
		double fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + fc*hi*vvi;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv - fc*hi*uui;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}
__global__ void updateEVSPH(double delta, double g, double yo, double ymax, double Radius, int * rightblk, int * topblk, double * blockyo, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	// Same as updateEV but with Spherical coordinate corrections
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;


	

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	double cm, fmu, fmv, y, phi, dphi,fmvp,fmup;

	double fc = pi / 21600.0; // 2*(2*pi/24/3600)


	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;


		//double yp= yo + yplus*delta / Radius*180.0 / pi;
		double yp;
		if (abs(blockyo[ibl] + (15.0 * delta / Radius*180.0 / pi) - ymax) < 1.0e-7)//if block is on the top side
		{
			//printf("Top Block\n");
			yp = blockyo[ibl] + (min(iy + 1, 15))*delta / Radius*180.0 / pi;
		}
		else
		{
			yp = blockyo[ibl] + (iy + 1)*delta / Radius*180.0 / pi;
		}
		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		fmup = 1.0;
		fmv = cos(phi);
		fmvp = cos(yp*pi/180.0);
		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0f;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		double cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		double dmdl = (fmup - fmu) / (cm*delta);// absurd even for spherical because fmu==1 always! What's up with that?
		double dmdt = (fmvp - fmv) / (cm*delta);
		double fG = vvi * dmdl - uui * dmdt;

		//With Coriolis
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + fc*sin(phi)*vvi;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv - fc*sin(phi)*uui;

		//without Coriolis
		//dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv;
		//dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}

__global__ void updateEVSPHATMUNI(double delta, double g, double yo, double ymax, double Radius, double uwind, float vwind, float Cd, int * rightblk, int * topblk, double * blockyo, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	// Same as updateEV but with Spherical coordinate corrections
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	double Uw = 0.0f;
	double Vw = 0.0f;


	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	double cm, fmu, fmv, y, phi, dphi, fmvp, fmup;

	double fc = pi / 21600.0; // 2*(2*pi/24/3600)

	Uw = uwind;// tex2D(texUWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);
	Vw = vwind;// tex2D(texVWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;


		//double yp= yo + yplus*delta / Radius*180.0 / pi;
		double yp;
		if (abs(blockyo[ibl] + (15.0 * delta / Radius*180.0 / pi) - ymax) < 1.0e-7)//if block is on the top side
		{
			yp = blockyo[ibl] + (min(iy + 1, 15))*delta / Radius*180.0 / pi;
		}
		else
		{
			yp = blockyo[ibl] + (iy + 1)*delta / Radius*180.0 / pi;
		}
		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		fmup = 1.0;
		fmv = cos(phi);
		fmvp = cos(yp*pi / 180.0);
		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0f;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		double cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		double dmdl = (fmup - fmu) / (cm*delta);// absurd even for spherical because fmu==1 always! What's up with that?
		double dmdt = (fmvp - fmv) / (cm*delta);
		double fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + 0.00121951*Cd*Uw*abs(Uw) + fc*sin(phi)*vvi; // why not fc*sin(phi)*hi*vvi ??
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv + 0.00121951*Cd*Vw*abs(Vw) - fc*sin(phi)*uui;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}

__global__ void updateEVSPHATM(double delta, double g, double yo, double ymax, double Radius, double xowind, double yowind, double dxwind, double Cd, int * rightblk, int * topblk, double * blockxo, double * blockyo, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	// Same as updateEV but with Spherical coordinate corrections
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	double Uw = 0.0f;
	double Vw = 0.0f;


	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	double cm, fmu, fmv,x, y, phi, dphi, fmvp, fmup;

	double fc = pi / 21600.0; // 2*(2*pi/24/3600)

	
	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = blockyo[ibl] + iy*delta / Radius*180.0 / pi;
		x = blockxo[ibl] + ix*delta / Radius*180.0 / pi;

		Uw = tex2D(texUWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);
		Vw = tex2D(texVWND, (x - xowind) / dxwind + 0.5, (y - yowind) / dxwind + 0.5);

		//double yp= yo + yplus*delta / Radius*180.0 / pi;
		double yp;
		if (abs(blockyo[ibl] + (15.0 * delta / Radius*180.0 / pi) - ymax) < 1.0e-7)//if block is on the top side
		{
			yp = blockyo[ibl] + (min(iy + 1, 15))*delta / Radius*180.0 / pi;
		}
		else
		{
			yp = blockyo[ibl] + (iy + 1)*delta / Radius*180.0 / pi;
		}
		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		fmup = 1.0;
		fmv = cos(phi);
		fmvp = cos(yp*pi / 180.0);
		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0f;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		double cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		double dmdl = (fmup - fmu) / (cm*delta);// absurd even for spherical because fmu==1 always! What's up with that?
		double dmdt = (fmvp - fmv) / (cm*delta);
		double fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) *cmdinv + 0.00121951*Cd*Uw*abs(Uw) + fc*sin(phi)*vvi; // why not fc*sin(phi)*hi*vvi ??
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) *cmdinv + 0.00121951*Cd*Vw*abs(Vw) - fc*sin(phi)*uui;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}

template <class T> __global__ void Advkernel( T dt, T eps, T * hh, T *zb, T *uu, T * vv, T *dh, T *dhu, T * dhv, T *zso, T *hho, T *uuo, T *vvo )
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	//int iright, itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//if (ix < nx && iy < ny)
	{

		T hold = hh[i];
		T ho, uo, vo;
		ho = hold + dt*dh[i];


		if (ho > eps) {
			//for (int l = 0; l < nl; l++) {
			//vector uo = vector(output[1 + dimension*l]);
			//vector ui = vector(input[1 + dimension*l]),
			//dhu = vector(updates[1 + dimension*l]);
			//foreach_dimension()
			uo = (hold*uu[i] + dt*dhu[i]) / ho;
			vo = (hold*vv[i] + dt*dhv[i]) / ho;
			//}


			//In the case of [multiplelayers](multilayer.h#viscous-friction-between-layers) we add the
			//viscous friction between layers.


		}
		else
		{// dry
			//for (int l = 0; l < nl; l++) {
			//vector uo = vector(output[1 + dimension*l]);
			//foreach_dimension()
			uo = T(0.0);
			vo = T(0.0);
		}


		zso[i] = zb[i] + ho;
		hho[i] = ho;
		uuo[i] = uo;
		vvo[i] = vo;
	}

}



template <class T> __global__ void cleanupGPU( T * hhi, T *zsi, T *uui, T *vvi, T * hho, T *zso, T *uuo, T *vvo)
{
	
	

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	//int ibl = blockIdx.x;
	//int i = ix + iy * 16 + ibl*blockDim.x;
	
	hho[i] = hhi[i];
	zso[i] = zsi[i];
	uuo[i] = uui[i];
	vvo[i] = vvi[i];
	
}

template <class T> __global__ void initdtmax( T epsi,T *dtmax)
{
	//int ix = blockIdx.x*blockDim.x + threadIdx.x;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	//int i = ix + iy*nx;
	int ix =  threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int i = ix + iy * 16 + ibl*(blockDim.x*blockDim.y);
	
		dtmax[i] = T(1.0) / epsi;
	
}



__global__ void minmaxKernel(int ntot, float *max, float *min, float *a) {
	__shared__ double maxtile[32];
	__shared__ double mintile[32];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ntot)
	{
		maxtile[tid] = a[i];
		mintile[tid] = a[i];
		__syncthreads();

		// strided index and non-divergent branch
		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			int index = 2 * s * tid;
			if (index < blockDim.x) {
				if (maxtile[tid + s] > maxtile[tid])
					maxtile[tid] = maxtile[tid + s];
				if (mintile[tid + s] < mintile[tid])
					mintile[tid] = mintile[tid + s];
			}
			__syncthreads();
		}

		if (tid == 0) {
			max[blockIdx.x] = maxtile[0];
			min[blockIdx.x] = mintile[0];
		}
	}
}

__global__ void finalminmaxKernel(float *max, float *min) {
	__shared__ double maxtile[32];
	__shared__ double mintile[32];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();

	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}
template <class T>
__global__ void resetdtmax( T *dtmax)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int i = ix + iy * 16 + ibl*(blockDim.x*blockDim.y);
	T initdt= T(1.0 / 1e-30);
	//if (ix < nx && iy < ny)
	{
		dtmax[i] = initdt;
	}
}

__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	extern __shared__ float sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	float mySum = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		mySum += g_idata[i + blockDim.x];

	sdata[tid] = mySum;
	__syncthreads();
	

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__global__ void reducemax3(float *g_idata, float *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	extern __shared__ float sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	float myMax = (i < n) ? g_idata[i] : -1e-30f;

	if (i + blockDim.x < n)
		myMax = max(myMax,g_idata[i + blockDim.x]);

	sdata[tid] = myMax;
	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = myMax = max(myMax, sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = myMax;
}

template <class T> __global__ void reducemin3(T *g_idata, T *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	T *sdata = SharedMemory<T>();
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	T myMin = (i < n) ? g_idata[i] : T(1e30);

	if (i + blockDim.x < n)
		myMin = min(myMin, g_idata[i + blockDim.x]);

	sdata[tid] = myMin;
	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = myMin = min(myMin, sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = myMin;
}


/*
template <unsigned int blockSize>
__global__ void reducemin6(int *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 1e30f;
	while (i < n) { sdata[tid] = min(g_idata[i], min( g_idata[i + blockSize],sdata[tid])); i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid + 256], sdata[tid]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid + 128], sdata[tid]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid + 64], sdata[tid]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] = min(sdata[tid + 32], sdata[tid]);
		if (blockSize >= 32) sdata[tid] = min(sdata[tid + 16], sdata[tid]);
		if (blockSize >= 16) sdata[tid] = min(sdata[tid + 8], sdata[tid]);
		if (blockSize >= 8) sdata[tid] = min(sdata[tid + 4], sdata[tid]);
		if (blockSize >= 4) sdata[tid] = min(sdata[tid + 2], sdata[tid]);
		if (blockSize >= 2) sdata[tid] = min(sdata[tid + 1], sdata[tid]);
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/

__global__ void leftdirichlet(int nybnd,float g,float dx,float xo,float ymax, float itime,int * rightblk, float *blockxo, float *blockyo, float *zs, float *zb, float *hh, float *uu, float *vv )
{
	
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//int xplus;
	//float hhi;
	float zsbnd;
	float itx = (blockyo[ibl]+iy*dx / ymax) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLZsBND, itime+0.5f, itx+0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(?) 
	if (abs(blockxo[ibl] - xo) <= 1.0e-16 && ix == 0 && zsbnd>zb[i])
	{
		//xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd-zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0f*(sqrtf(g*max(hh[iright], 0.0f)) - sqrtf(g*max(zsbnd - zb[iright], 0.0f))) + uu[iright];
		vv[i] = 0.0f;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
	}
}

template <class T> __global__ void DRYBND(int isright, int istop, T eps, T *zb, T *zs,T *hh,T *uu, T *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	// left || right || bot || top
	if ((isright < 0 && ix == 0) || (isright > 0 && ix == 15) || (istop < 0 && iy == 0) || (istop > 0 && iy == 15))
	{
		uu[i] = (T) 0.0;
		vv[i] = (T) 0.0;
		hh[i] = eps;
		zs[i] = zb[i] + eps;

	}
	

}

template <class T> __global__ void dirichlet(int isright, int istop, int nbnd, T g, T dx, T xo, T xmax, T yo, T ymax, T itime, int * bndblk, int * neighbourblk, T *blockxo, T *blockyo, T *zs, T *zb, T *hh, T *un, T *ut)
{

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ib = blockIdx.x;


	//printf("ibl=%d\n", bndblk[ib]);
	int ibl = bndblk[ib];

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	int inside;


	T xx, yy;
	int bnd, bnd_c;
	T sign, umean;
	float itx;

	sign = T(isright) + T(istop);




	//int xplus;
	//float hhi;
	float zsbnd;
	T zsinside;

	xx = blockxo[ibl] + ix*dx;
	yy = blockyo[ibl] + iy*dx;

	//if (ix == 0 && iy == 0)
	//{
	//	printf("ib=%d\tibl=%d\txx=%f\tyy=%f\n", ib, ibl, xx, yy);
	//}
	if (isright < 0)
	{
		inside = findrightG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!
		itx = (yy - yo) / (ymax - yo)*nbnd;
		zsbnd = tex2D(texLZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
		
		//if (ix == 0 && iy == 0)
		//{
		//	printf("yy=%f\titx=%f\tzsbnd=%f\n", yy, itx, zsbnd);
		//}
		//printf("yy=%f\titx=%f\tzsbnd=%f\n", yy, itx, zsbnd);
	}
	else if (isright > 0)
	{
		inside = findleftG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!
		itx = (yy - yo) / (ymax - yo)*nbnd;
		zsbnd = tex2D(texRZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}
	else if (istop < 0)//isright must be ==0!
	{
		inside = findtopG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texBZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}
	else // istop ==1 && isright ==0
	{
		inside = findbotG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texTZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}

	if (bnd == bnd_c && zsbnd>zb[i])
	{
		zsinside = zs[inside];
		un[i] =  sign*T(2.0)*(sqrt(g*max(hh[inside], T(0.0))) - sqrt(g*max(zsbnd - zb[inside], T(0.0)))) + un[inside];
		ut[i] = T(0.0);
		zs[i] = zsinside;
		//ut[i] = ut[inside];
		hh[i] = hh[inside];
	}


	


	/*

	int iright;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//int xplus;
	//float hhi;
	float zsbnd;
	float itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(?) 
	if (abs(blockxo[ibl] - xo) <= 1.0e-16 && ix == 0 && zsbnd>zb[i])
	{
		//xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0f*(sqrtf(g*max(hh[iright], 0.0f)) - sqrtf(g*max(zsbnd - zb[iright], 0.0f))) + uu[iright];
		vv[i] = 0.0f;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
	}

	*/
}

template <class T> __global__ void ABS1D(int isright, int istop,int nbnd, T g, T dx, T xo, T yo, T xmax,T ymax, T itime, int * bndblck, int * neighbourblk, T *blockxo, T *blockyo, T *zs, T *zb, T *hh, T *un, T *ut)
{

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ib = blockIdx.x;

	int ibl = bndblck[ib];
	
	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	int inside;

	// left bnd: isrigit = -1; istop=0;
	// right bnd: isright = 1; istop=0;
	// bottom bnd: isright = 0; istop=-1;
	// top bnd: isright = 0; istop=1;

	T xx,yy;
	int bnd, bnd_c;
	T  sign,umean;
	float itx;
	
	sign = T(isright) + T(istop);




	//int xplus;
	//float hhi;
	float zsbnd;
	T zsinside;

	xx = blockxo[ibl] + ix*dx;
	yy = blockyo[ibl] + iy*dx;


	if (isright < 0)
	{
		inside= findrightG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!

		itx = (yy - yo) / (ymax - yo)*nbnd;

		zsbnd = tex2D(texLZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??) 
	}
	else if (isright > 0)
	{
		inside = findleftG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!
		itx = (yy - yo) / (ymax - yo)*nbnd;
		zsbnd = tex2D(texRZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}
	else if (istop < 0)//isright must be ==0!
	{
		inside= findtopG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texBZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}
	else // istop ==1 && isright ==0
	{
		inside = findbotG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texTZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
	}

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);


	
	
	
	
	umean = T(0.0);
	
	
	if (bnd == bnd_c && zsbnd>zb[i])
	{
		zsinside = zs[inside];
		//xplus = min(ix + 1, nx - 1);
		//hh[i] = zsbnd - zb[i];
		//zs[i] = zsbnd;
		//uu[i] = -2.0f*(sqrtf(g*max(hh[iright], 0.0f)) - sqrtf(g*max(zsbnd - zb[iright], 0.0f))) + uu[iright];
		//vv[i] = 0.0f;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
		//printf("zsbnd=%f\n", zsbnd);
		un[i] = sign*sqrt(g / hh[i])*(zsinside - zsbnd)+umean;
		zs[i] = zsinside;
		ut[i] = ut[inside];
		hh[i] = hh[inside];
	}
}

template <class T> __global__ void ABS1DNEST(int isright, int istop, int nbnd, T g, T dx, T xo, T yo, T xmax, T ymax, T itime, int * bndblck, int * neighbourblk, T *blockxo, T *blockyo, T *zs, T *zb, T *hh, T *un, T *ut)
{

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ib = blockIdx.x;

	int ibl = bndblck[ib];

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	int inside;

	// left bnd: isrigit = -1; istop=0;
	// right bnd: isright = 1; istop=0;
	// bottom bnd: isright = 0; istop=-1;
	// top bnd: isright = 0; istop=1;

	T xx, yy;
	int bnd, bnd_c;
	T  sign;
	float itx;

	sign = T(isright) + T(istop);




	//int xplus;
	//float hhi;
	float zsbnd;
	float unbnd=0.0;
	float utbnd=0.0;

	T zsinside;

	xx = blockxo[ibl] + ix*dx;
	yy = blockyo[ibl] + iy*dx;


	if (isright < 0) // left bnd
	{
		inside = findrightG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!

		itx = (yy - yo) / (ymax - yo)*nbnd;

		zsbnd = tex2D(texLZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??) 
		unbnd = tex2D(texLUBND, itime + 0.5f, itx + 0.5f);
		utbnd = tex2D(texLVBND, itime + 0.5f, itx + 0.5f);

		
	}
	else if (isright > 0) // right bnd
	{
		inside = findleftG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = ix;
		//itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nbnd - 1.0f));//Bleark!
		itx = (yy - yo) / (ymax - yo)*nbnd;
		zsbnd = tex2D(texRZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
		unbnd = tex2D(texRUBND, itime + 0.5f, itx + 0.5f);
		utbnd = tex2D(texRVBND, itime + 0.5f, itx + 0.5f);
		

	}
	else if (istop < 0) // bottom bnd  (isright must be ==0!)
	{
		inside = findtopG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texBZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
		unbnd = tex2D(texBVBND, itime + 0.5f, itx + 0.5f);
		utbnd = tex2D(texBUBND, itime + 0.5f, itx + 0.5f);

	}
	else // top bnd istop ==1 && isright ==0
	{
		inside = findbotG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = iy;
		//itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nbnd - 1.0f));
		itx = (xx - xo) / (xmax - xo)*nbnd;
		zsbnd = tex2D(texTZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(is this totally sure??)
		unbnd = tex2D(texTVBND, itime + 0.5f, itx + 0.5f);
		utbnd = tex2D(texTUBND, itime + 0.5f, itx + 0.5f);
	}

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);






	


	if (bnd == bnd_c && zsbnd>zb[i])
	{
		zsinside = zs[inside];
		//xplus = min(ix + 1, nx - 1);
		//hh[i] = zsbnd - zb[i];
		//zs[i] = zsbnd;
		//uu[i] = -2.0f*(sqrtf(g*max(hh[iright], 0.0f)) - sqrtf(g*max(zsbnd - zb[iright], 0.0f))) + uu[iright];
		//vv[i] = 0.0f;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t uubnd=%f\t", zsbnd,umean);
		//}
		//printf("zsbnd=%f\n", zsbnd);
		un[i] = sign*sqrt(g / hh[i])*(zsinside - zsbnd) + T(unbnd);
		zs[i] = zsinside;
		ut[i] = T(utbnd);//ut[inside];
		hh[i] = hh[inside];
	}
}

template <class T> __global__ void noslipbnd(int isright, int istop, int * bndblck, int * neighbourblk, T *zs, T *hh, T *un)
{
	//

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ib = blockIdx.x;

	int ibl = bndblck[ib];

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	int inside;

	int bnd, bnd_c;

	if (isright < 0)
	{
		inside = findrightG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = ix;
		
	}
	else if (isright > 0)
	{
		inside = findleftG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = ix;
		
	}
	else if (istop < 0)//isright must be ==0!
	{
		inside = findtopG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 0;
		bnd = iy;
		
	}
	else // istop ==1 && isright ==0
	{
		inside = findbotG(ix, iy, neighbourblk[ibl], ibl, blockDim.x);
		bnd_c = 15;
		bnd = iy;
		
	}

	if (bnd == bnd_c)
	{
		 
		
		un[i] = T(0.0);
		zs[i] = zs[inside];
		//ut[i] = ut[inside];
		hh[i] = hh[inside];
	}

}
__global__ void leftdirichletD(int nybnd, double g, double dx, double xo, double ymax, double itime, int * rightblk, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);




	int iright;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//float hhi;
	float zsbnd; //remains a float because this is how it is stored on the texture memory // I don't think it is a big deal
	float itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLZsBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(?) 
	if (abs(blockxo[ibl] - xo) <= 1.0e-16 && ix == 0 && zsbnd>zb[i])
	{
		//xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0*(sqrt(g*max(hh[iright], 0.0)) - sqrt(g*max(zsbnd - zb[iright], 0.0))) + uu[iright];
		vv[i] = 0.0;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
	}
}


__global__ void rightdirichlet( int nybnd, float g, float dx, float xmax, float ymax, float itime,int *leftblk,float *blockxo, float *blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	
	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//int xminus;
	//float hhi;
	float zsbnd;
	float itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texRZsBND, itime+0.5f, itx+0.5f);

	if (abs(blockxo[ibl] + 15 * dx - xmax) <= 1.0e-16 && ix == 15 && zsbnd>zb[i])
	{
		//xminus = max(ix - 1, 0);
		//printf("zsbnd=%f\n", zsbnd);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = +2.0f*(sqrtf(g*max(hh[ileft], 0.0f)) - sqrtf(g*max(zsbnd - zb[ileft], 0.0f))) + uu[ileft];
		vv[i] = 0.0f;
	}
}

__global__ void rightdirichletD( int nybnd, double g,double dx,double xmax,double ymax, double itime, int*leftblk, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ileft;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

	//int xminus;
	//float hhi;
	float zsbnd;
	float itx = (blockyo[ibl] + iy*dx / ymax) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texRZsBND, itime + 0.5f, itx + 0.5f);
	if (abs(blockxo[ibl] + 15 * dx - xmax) <= 1.0e-16 && ix == 15 && zsbnd>zb[i])
	{
		//xminus = max(ix - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = +2.0*(sqrt(g*max(hh[ileft], 0.0)) - sqrt(g*max(zsbnd - zb[ileft], 0.0))) + uu[ileft];
		vv[i] = 0.0f;
	}
}

__global__ void topdirichlet( int nxbnd, float g,float dx, float xmax, float ymax, float itime,int * botblk, float *blockxo, float *blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int yminus;
	//float hhi;
	float zsbnd;
	float itx = (blockxo[ibl]+ix*dx / xmax) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texTZsBND, itime + 0.5f, itx + 0.5f);
	if (abs(blockyo[ibl]+15*dx-ymax)<=1.0e-16 && iy == 15 && zsbnd>zb[i])
	{
		//yminus = max(iy - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = +2.0f*(sqrtf(g*max(hh[ibot], 0.0f)) - sqrtf(g*max(zsbnd - zb[ibot], 0.0f))) + vv[ibot];
		uu[i] = 0.0f;
	}
}

__global__ void topdirichletD( int nxbnd, double g,double dx,double xmax,double ymax, double itime,int * botblk,double *blockxo, double *blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int yminus;
	//float hhi;
	float zsbnd;
	float itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texTZsBND, itime + 0.5f, itx + 0.5f);
	if (abs(blockyo[ibl] + 15 * dx - ymax) <= 1.0e-16 && iy == 15 && zsbnd>zb[i])
	{
		//yminus = max(iy - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = +2.0*(sqrt(g*max(hh[ibot], 0.0)) - sqrt(g*max(zsbnd - zb[ibot], 0.0))) + vv[ibot];
		uu[i] = 0.0;
	}
}
__global__ void botdirichlet( int nxbnd, float g, float dx,float xmax, float yo, float itime,int * topblk,float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int yplus;
	//float hhi;
	float zsbnd;
	float itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texBZsBND, itime + 0.5f, itx + 0.5f);
	if (abs(blockyo[ibl] - yo) <= 1.0e-16 && iy == 0 && zsbnd>zb[i])
	{
		//yplus = min(iy + 1, ny-1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = -2.0f*(sqrtf(g*max(hh[itop], 0.0f)) - sqrtf(g*max(zsbnd - zb[itop], 0.0f))) + vv[itop];
		uu[i] = 0.0f;
	}
}

__global__ void botdirichletD( int nxbnd, double g,double dx,double xmax,double yo, double itime,int * topblk,double * blockxo, double * blockyo, double*zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int yplus;
	//float hhi;
	float zsbnd;
	float itx = (blockxo[ibl] + ix*dx / xmax) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texBZsBND, itime + 0.5f, itx + 0.5f);
	if (abs(blockyo[ibl] - yo) <= 1.0e-16 && iy == 0 && zsbnd>zb[i])
	{
		//yplus = min(iy + 1, ny - 1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = -2.0*(sqrt(g*max(hh[itop], 0.0)) - sqrt(g*max(zsbnd - zb[itop], 0.0))) + vv[itop];
		uu[i] = 0.0;
	}
}



template <class T> __global__ void bottomfriction(int smart, T dt,T eps, T* cf, T *hh, T *uu, T *vv)
{
	// Shear stress equation:
	// Taub=cf*rho*U*sqrt(U^2+V^2)
	// uu=uu-dt*(Tub/(rho*h))
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	//int itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	
	T normu,hhi,uui,vvi;
	T ee = T(2.71828182845905);
	//if (ix < nx && iy < ny)
	{

		hhi = hh[i];
		uui = uu[i];
		vvi = vv[i];
		if (hhi > eps)
		{
			normu = sqrt(uui * uui + vvi * vvi);
			//T frc = (T(1.0) + dt*cf*(normu) / hhi);
				//u.x[] = h[]>dry ? u.x[] / (1 + dt*cf*norm(u) / h[]) : 0.;
			//uu[i] = uui / frc;
			//vv[i] = vvi / frc;
			T cfi = cf[i];
			if (smart == 1)//Smart friction formulation
			{
				T zo = cfi;
				T Hbar = hhi / zo;
				if (Hbar <= ee)
				{
					cfi = T(1.0) / (T(0.46)*Hbar);
				}
				else
				{
					cfi = T(1.0)/(T(2.5)*(log(Hbar) - T(1.0) + T(1.359)/Hbar));
				}
				cfi = cfi*cfi; // 
			}
			if (smart == -1)// Manning friction formulation
			{
				T n = cfi;
				cfi = T(9.81)*n*n / cbrt(hhi);

			}

			T tb = cfi*normu/hhi*dt;
			uu[i] = uui / (T(1.0)+tb);
			vv[i] = vvi / (T(1.0)+tb);
		}
		
	}

}


__global__ void noslipbndall( float dt, float eps,int * leftblk,int*rightblk,int *topblk, int *botblk, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ileft, iright,itop,ibot;

	ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//float normu, hhi;

	//if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		if (leftblk[ibl] == ibl )
		{
			uu[i] = 0.0f;
			zs[i] = zs[iright];
			hh[i] = max(zs[iright]-zb[i],eps);
		}
		if ( rightblk[ibl]== ibl)
		{
			uu[i] = 0.0f;
			zs[i] = zs[ileft];
			hh[i] = max(zs[ileft] - zb[i], eps);

		}

		if ( botblk[ibl] == ibl )
		{
			vv[i] = 0.0f;
			zs[i] = zs[itop];
			hh[i] = max(zs[itop] - zb[i], eps);
		}
		if ( topblk[ibl] == ibl)
		{
			vv[i] = 0.0f;
			zs[i] = zs[ibot];
			hh[i] = max(zs[ibot] - zb[i], eps);

		}

	}

}
template <class T> __global__ void noslipbndLeft(T xo, T eps, int * rightblk,T* blockxo, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int iright;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int  xplus;
	T zsp;

	if (blockxo[ibl]==xo && ix==0)
	{
		//xplus = min(ix + 1, nx - 1);
		iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);

		zsp = zs[iright];

		
		uu[i] = T(0.0);
		zs[i] = zsp;
		hh[i] = max(zsp - zb[i], eps);
		
		

	}

}



template <class T> __global__ void noslipbndBot(T yo, T eps,int * topblk,T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int itop;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int yplus;
	
	if (blockyo[ibl]==yo && iy==0)
	{
		//yplus = min(iy + 1, ny - 1);
		itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
		vv[i] = T(0.0);
		zs[i] = zs[itop];
		hh[i] = max(zs[itop] - zb[i], eps);



	}

}


template <class T> __global__ void noslipbndRight(T dx, T xmax, T eps,int *leftblk,T* blockxo, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ileft;

	
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	//ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);
	//int xminus;
	

	if ((blockxo[ibl]+15*dx) == xmax && ix==15)
	{
		//xminus = max(ix - 1, 0);
		ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
		uu[i] = T(0.0);
		zs[i] = zs[ileft];
		hh[i] = max(zs[ileft] - zb[i], eps);



	}

}
template <class T> __global__ void noslipbndTop(T dx, T ymax, T eps, int * botblk, T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	int ibot;

	//ileft = findleftG(ix, iy, leftblk[ibl], ibl, blockDim.x);
	//iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	//itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	
	

	if (blockyo[ibl]+15*dx == ymax && iy==15)
	{
		//yminus = max(iy - 1, 0);
		ibot = findbotG(ix, iy, botblk[ibl], ibl, blockDim.x);

		vv[i] = T(0.0);
		zs[i] = zs[ibot];
		hh[i] = max(zs[ibot] - zb[i], eps);



	}

}

template <class T>
__global__ void storeTSout(int noutnodes, int outnode, int istep, int inode,int jnode, int blknode, T *zs, T *hh, T *uu, T *vv,T * store)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	

	if (ibl == blknode && ix == inode && iy == jnode )
	{
		store[0 + outnode * 4 + istep*noutnodes * 4] = hh[i];
		store[1 + outnode * 4 + istep*noutnodes * 4] = zs[i];
		store[2 + outnode * 4 + istep*noutnodes * 4] = uu[i];
		store[3 + outnode * 4 + istep*noutnodes * 4] = vv[i];
	}
}


template <class T>
__global__ void addavg_var( T * Varmean, T * Var)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	__shared__ T mvari[16][16];
	__shared__ T vari[16][16];

	//if (ix < nx && iy < ny)
	{

		mvari[ix][iy] = Varmean[i];
		vari[ix][iy] = Var[i];

		Varmean[i] = mvari[ix][iy] + vari[ix][iy];
	}


}

template <class T>
__global__ void divavg_var( T ntdiv, T * Varmean)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);
	__shared__ T mvari[16][16];
	//if (ix < nx && iy < ny)
	{
		mvari[ix][iy] = Varmean[i];
		Varmean[i] = mvari[ix][iy] / ntdiv;
	}


}

template <class T>
__global__ void resetavg_var( T * Varmean)
{
	/*
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		Varmean[i] = T(0.0);
	}
	*/
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int i = ix + iy * 16 + ibl*(blockDim.x*blockDim.y);
	Varmean[i] = T(0.0);
}

template <class T>
__global__ void resetmax_var( T * Varmax)
{
	/*
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		Varmax[i] = T(-1.0/epsilone);
	}
	*/
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int i = ix + iy * 16 + ibl*(blockDim.x*blockDim.y);
	Varmax[i] = T(-1.0 / epsilone);
}

template <class T>
__global__ void max_var(T * Varmax, T * Var)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	__shared__ T mvari[16][16];
	__shared__ T vari[16][16];

	//if (ix < nx && iy < ny)
	{

		mvari[ix][iy] = Varmax[i];
		vari[ix][iy] = Var[i];

		Varmax[i] = max(mvari[ix][iy], vari[ix][iy]);
	}


}

template <class T>
__global__ void CalcVorticity( T * Vort,T * dvdx, T * dudy)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	__shared__ T dvdxi[16][16];
	__shared__ T dudyi[16][16];

	//if (ix < nx && iy < ny)
	{

		dvdxi[ix][iy] = dvdx[i];
		dudyi[ix][iy] = dudy[i];

		Vort[i] = dvdxi[ix][iy] - dudyi[ix][iy];
	}


}

template <class T>
__global__ void discharge_bnd_v(T xstart,T xend, T ystart,T yend, T dx, T dt, T qnow, T disarea, int * Riverblks, T * blockxo, T * blockyo, T* zs, T* hh)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = Riverblks[blockIdx.x];




	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	T xx = blockxo[ibl] + ix*dx;
	T yy = blockyo[ibl] + iy*dx;

	if (xx >= xstart && xx <= xend && yy >= ystart && yy <= yend)
	{
		T dzsdt = qnow*dt / disarea;
		zs[i] = zs[i] + dzsdt;
		// Do hh[i] too although Im not sure it is worth it
		hh[i] = hh[i] + dzsdt;
	}
	


}

// template not needed here
template <class T> 
__global__ void Rain_on_grid(double maskzs, double xorain, double yorain, double dxrain, double delta, double*blockxo, double *blockyo, double dt,  T * zs, T *hh)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	T x = blockxo[ibl] + ix*delta;
	T y = blockyo[ibl] + iy*delta;

	double Rainhh;
	T zzi = zs[i];
	T hhi = hh[i];



	if (zzi < maskzs)
	{
		//printf("%f\n", zzi);
		Rainhh = tex2D(texRAIN, (x - xorain) / dxrain + 0.5, (y - yorain) / dxrain + 0.5);
		//Rainhh = 1 * (xorain) / dxrain + 0.5;
		Rainhh = Rainhh / 1000.0 / 3600.0*dt; // convert from mm/hrs to m/s and to cumulated rain height within a step

		zs[i] = Rainhh + zzi;
		hh[i] = Rainhh + hhi;
	}
}

template <class T>
__global__ void Rain_on_gridUNI(double maskzs, double rainuni, double dt, T * zs, T *hh)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	
	double Rainhh;
	
	double zzi = zs[i];
	double hhi = hh[i];


	//Rainhh = tex2D(texRAIN, (x - xorain) / dxrain + 0.5, (y - yorain) / dxrain + 0.5);
	Rainhh = rainuni / 1000.0 / 3600.0*dt; // convert from mm/hrs to m/s and to cumulated rain height within a step
	if (zzi < maskzs)
	{
		//printf("%f\n", Rainhh);
		zs[i] = Rainhh + zzi;
		hh[i] = Rainhh + hhi;
	}

}

/*
__global__ void discharge_bnd_h(int nx, int ny, DECNUM dx, DECNUM eps, DECNUM qnow, int istart, int jstart, int iend, int jend, DECNUM *hu, DECNUM *hv, DECNUM *qx, DECNUM *qy, DECNUM * uu, DECNUM *vv)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;

	if (ix >= istart && ix <= iend &&iy >= jstart && iy <= jend)
	{
		//discharge should run along the x or y axis
		//TODO modify the algorithm to alow diagonal discharges
		if (istart == iend)//discharge on uu along a vertical line
		{
			float A = 0.0;

			for (int k = jstart; k <= jend; k++)
			{
				A = A + (cbrtf(hu[ix + k*ny])*dx);
			}

			float cst = qnow / max(A, eps);
			uu[i] = cst*sqrtf(hu[i]);
			qx[i] = cst*cbrtf(hu[i]);
		}

		if (jstart == jend)//discharge on uu along a vertical line
		{
			float A = 0.0;

			for (int k = istart; k <= iend; k++)
			{
				A = A + (cbrtf(hv[k + iy*ny])*dx);
			}

			float cst = qnow / max(A, eps);
			vv[i] = cst*sqrt(hv[i]);
			qy[i] = cst*cbrtf(hv[i]);
		}




	}



}
*/


template <class T>
__global__ void NextHDstep(int nx, int ny, T * Uold, T * Unew)
{
	//int ix = blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;


	if (ix<nx && iy<ny)
	{
		Uold[ix + iy*nx] = Unew[ix + iy*nx];
	}
}

template <class T>
__global__ void HD_interp(int nx, int ny, int backswitch, int nhdstp, T totaltime, T hddt, T * Uold, T * Unew, T * UU)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T Uxo[16][16];
	__shared__ T Uxn[16][16];
	//	__shared__ float Ums[16];


	T fac =(T) 1.0;
	/*Ums[tx]=Umask[ix];*/


	if (backswitch>0)
	{
		fac = (T)-1.0;
	}


	if (ix<nx && iy<ny)
	{
		Uxo[tx][ty] = fac*Uold[ix + nx*iy]/**Ums[tx]*/;
		Uxn[tx][ty] = fac*Unew[ix + nx*iy]/**Ums[tx]*/;

		UU[ix + nx*iy] = Uxo[tx][ty] + (totaltime - hddt*nhdstp)*(Uxn[tx][ty] - Uxo[tx][ty]) / hddt;
	}
}

template <class T>
__global__ void Deform(T scale, T * def, T * zs, T * zb)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int i = ix + iy * blockDim.x + ibl*(blockDim.x*blockDim.y);

	zs[i] = zs[i] + def[i] * scale;
	zb[i] = zb[i] + def[i] * scale;
	


}

