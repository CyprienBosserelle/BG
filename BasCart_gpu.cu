﻿//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2017 Bosserelle                                                 //
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

// includes, system


#include "Header.cuh"



double phi = (1.0f + sqrt(5.0f)) / 2;
double aphi = 1 / (phi + 1);
double bphi = phi / (phi + 1);
double twopi = 8 * atan(1.0f);

double g = 1.0;// 9.81;
double rho = 1025.0;
double eps = 0.0001;
double CFL = 0.5;

double totaltime = 0.0;


double dt, dx;
int nx, ny;

double delta;

double *x, *y;
double *x_g, *y_g;

double *zs, *hh, *zb, *uu, *vv;
double *zso, *hho, *uuo, *vvo;


double * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
double *dzsdx, *dzsdy;
//double *fmu, *fmv;

double *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;

double * Fhu, *Fhv;

double * dh, *dhu, *dhv;

double dtmax = 1.0 / epsilon;


void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}


// Main loop that actually runs the model
void mainloopGPU()
{
	
	

	
	
}






void updateGPU(int nx, int ny, double dt, double eps)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);


	//__global__ void MetricTerm(int nx, int ny, double delta, double G, double *h, double *u, double *v, double * fmu, double * fmv, double* dhu, double *dhv, double *Su, double *Sv, double * Fqux, double * Fquy, double * Fqvx, double * Fqvy)
	//MetricTerm << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hh_g, uu_g, vv_g, fmu_g, fmv_g, dhu_g, dhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g);

}


void advanceGPU(int nx, int ny, double dt, double eps)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	
}



int main(int argc, char **argv)
{

	
	//Model starts Here//

	//The main function setups all the init of the model and then calls the mainloop to actually run the model


	//First part reads the inputs to the model 
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Prepare output file
	// Run main loop
	// Clean up and close


	// Start timer to keep track of time 
	clock_t startcputime, endcputime;


	startcputime = clock();



	// This is just for temporary use
	nx = 32;
	ny = 32;
	double length = 1.0;
	delta = length / nx;


	double *xx, *yy;
	dt = 0.0;// Will be resolved in update




	hh = (double *)malloc(nx*ny * sizeof(double));
	uu = (double *)malloc(nx*ny * sizeof(double));
	vv = (double *)malloc(nx*ny * sizeof(double));
	zs = (double *)malloc(nx*ny * sizeof(double));
	zb = (double *)malloc(nx*ny * sizeof(double));

	hho = (double *)malloc(nx*ny * sizeof(double));
	uuo = (double *)malloc(nx*ny * sizeof(double));
	vvo = (double *)malloc(nx*ny * sizeof(double));
	zso = (double *)malloc(nx*ny * sizeof(double));

	dhdx = (double *)malloc(nx*ny * sizeof(double));
	dhdy = (double *)malloc(nx*ny * sizeof(double));
	dudx = (double *)malloc(nx*ny * sizeof(double));
	dudy = (double *)malloc(nx*ny * sizeof(double));
	dvdx = (double *)malloc(nx*ny * sizeof(double));
	dvdy = (double *)malloc(nx*ny * sizeof(double));

	dzsdx = (double *)malloc(nx*ny * sizeof(double));
	dzsdy = (double *)malloc(nx*ny * sizeof(double));


	//fmu = (double *)malloc(nx*ny * sizeof(double));
	//fmv = (double *)malloc(nx*ny * sizeof(double));
	Su = (double *)malloc(nx*ny * sizeof(double));
	Sv = (double *)malloc(nx*ny * sizeof(double));
	Fqux = (double *)malloc(nx*ny * sizeof(double));
	Fquy = (double *)malloc(nx*ny * sizeof(double));
	Fqvx = (double *)malloc(nx*ny * sizeof(double));
	Fqvy = (double *)malloc(nx*ny * sizeof(double));
	Fhu = (double *)malloc(nx*ny * sizeof(double));
	Fhv = (double *)malloc(nx*ny * sizeof(double));

	dh = (double *)malloc(nx*ny * sizeof(double));
	dhu = (double *)malloc(nx*ny * sizeof(double));
	dhv = (double *)malloc(nx*ny * sizeof(double));

	//x = (double *)malloc(nx*ny * sizeof(double));
	xx = (double *)malloc(nx * sizeof(double));
	//y = (double *)malloc(nx*ny * sizeof(double));
	yy = (double *)malloc(ny * sizeof(double));

	//init variables
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			zb[i + j*nx] = 0.0;
			uu[i + j*nx] = 0.0;
			vv[i + j*nx] = 0.0;
			//x[i + j*nx] = (i-nx/2)*delta+0.5*delta;
			xx[i] = (i - nx / 2)*delta + 0.5*delta;
			yy[j] = (j - ny / 2)*delta + 0.5*delta;
			//y[i + j*nx] = (j-ny/2)*delta + 0.5*delta;
			//fmu[i + j*nx] = 1.0;
			//fmv[i + j*nx] = 1.0;
		}
	}

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			double a;

			a = sq(xx[i]) + sq(yy[j]);
			//b =x[i + j*nx] * x[i + j*nx] + y[i + j*nx] * y[i + j*nx];


			//if (abs(a - b) > 0.00001)
			//{
			//	printf("%f\t%f\n", a, b);
			//}



			hh[i + j*nx] = 0.1 + 1.*exp(-200.*(a));

			zs[i + j*nx] = zb[i + j*nx] + hh[i + j*nx];
		}
	}


	create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	//while (totaltime < 10.0)
	for (int i = 0; i <10; i++)
	{
		mainloopCPU();
		totaltime = totaltime + dt;
		write2varnc(nx, ny, totaltime, hh);
		//write2varnc(nx, ny, totaltime, dhdx);
	}






	endcputime = clock();
	printf("End Computation totaltime=%f\n", totaltime);
	printf("Total runtime= %d  seconds\n", (endcputime - startcputime) / CLOCKS_PER_SEC);











}

