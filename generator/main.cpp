#define FOR_RELEASE

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <assert.h>
#include <string.h>

static const double UPPER = 0.95;
static const double LOWER = 0.05;

#ifndef RADIUS_BOUND
#define RADIUS_BOUND 0.05
#endif

#define DOMAIN 1

using namespace std;

double RandomEqual(double min,double max)
{
	double x = (double)rand()/RAND_MAX;
	return x*(max-min)+min;
}

double RandomPeak(double min,double max,int dim)
{
	double sum = 0.0;
	for (int d=0; d<dim; d++) sum += RandomEqual(0,1);
	sum /= dim;
	return sum*(max-min)+min;
}

double RandomNormal(double med,double var)
{
	return RandomPeak(med-var,med+var,12);
}

double loc(double v, double s)
{
	double d = v / s;
	long i = (long)d;
	if(d > i)
	{
		i ++;	
	}

	return (double)i * s;
}

// void GenerateDataEqually(ostream& dou_data,int Count,int DupRatio,int Dimensions)
void GenerateDataEqually(double** center, int Dimensions, int Count, int DupRatio)
{
	double step = 1.0 / ((double)Count / (double)DupRatio);

	for (int i=0; i<Count; i++)
	{
		double x[Dimensions];
		for (int d=0; d<Dimensions; d++)
		{
// 			x[d] = RandomEqual(0,1);
			x[d] = RandomEqual(LOWER, UPPER);
			if(DupRatio == 1)
			{
// 				dou_data << x[d] << " ";
				center[i][d] = x[d];
			}
			else
			{
// 				dou_data << loc(x[d],step) << " ";
				center[i][d] = loc(x[d],step);
			}
		}
// 		dou_data << endl;
	}
}

// void GenerateDataCorrelated(ostream& dou_data,int Count,int DupRatio,int Dimensions)
void GenerateDataCorrelated(double** center, int Dimensions, int Count, int DupRatio)
{
	double step = 1.0 / ((double)Count / (double)DupRatio);

	double x[Dimensions];
	for (long i=0; i<Count; i++) {
again:
// 		double v = RandomPeak(0,1,Dimensions);
		double v = RandomPeak(LOWER, UPPER, Dimensions);
		for (int d=0; d<Dimensions; d++) x[d] = v;
// 		double l = v<=0.5 ? v:1.0-v;
		double l = v<=(LOWER+UPPER)/2 ? v : UPPER-v;
		for (int d=0; d<Dimensions; d++) {
// 			double h = RandomNormal(0,l);
			double h = RandomNormal(LOWER, l);
			x[d] += h;
			x[(d+1)%Dimensions] -= h;
		}
// 		for (int d=0; d<Dimensions; d++) if (x[d]<0 || x[d]>=1) goto again;
		for (int d=0; d<Dimensions; d++) if (x[d]<LOWER || x[d]>=UPPER) goto again;
		
		for (int d=0; d<Dimensions; d++) {
			if(DupRatio == 1)
			{
// 				dou_data << x[d] << " ";
				center[i][d] = x[d];
			}
			else
			{
// 				dou_data << loc(x[d],step) << " ";
				center[i][d] = loc(x[d],step);
			}
		}
// 		dou_data << endl;
	}
}

// void GenerateDataAnticorrelated(ostream& dou_data,int Count,int DupRatio,int Dimensions)
void GenerateDataAnticorrelated(double** center, int Dimensions, int Count, int DupRatio)
{
	double step = 1.0 / ((double)Count / (double)DupRatio);

	double x[Dimensions];
	for (long i=0; i<Count; i++) 
	{
again:
// 		double v = RandomNormal(0.5,0.25);
		double v = RandomNormal((LOWER+UPPER)/2, (LOWER+UPPER)/4);
		for (int d=0; d<Dimensions; d++) x[d] = v;
// 		double l = v<=0.5 ? v:1.0-v;
		double l = v<=(LOWER+UPPER)/2 ? v : UPPER-v;
		for (int d=0; d<Dimensions; d++) 
		{
			double h = RandomEqual(-l,l);
			x[d] += h;
			x[(d+1)%Dimensions] -= h;
		}
		for (int d=0; d<Dimensions; d++) if (x[d]<LOWER || x[d]>=UPPER) goto again;
		for (int d=0; d<Dimensions; d++) 
		{
			if(DupRatio == 1)
			{
// 				dou_data << x[d] << " ";
				center[i][d] = x[d];
			}
			else
			{
// 				dou_data << loc(x[d],step) << " ";
				center[i][d] = loc(x[d],step);
			}
		}
// 		dou_data << endl;
	}
}

// void GenerateData(int Dimensions,char Distribution,int Count,int DupRatio, 
// 									ostream& dou_data)
void GenerateCenter(double** center, int Dimensions, char Distribution, int Count)
{
 	if (Count <= 0) {
 		cout << "Amount should be greater than 0" << endl;
 		return;
 	}
	if (Dimensions < 2) {
		cout << "Dimension should be greater than 2" << endl;
		return;
	}
	switch (Distribution) {
		case 'E':
		case 'e': Distribution = 'E'; break;
		case 'C':
		case 'c': Distribution = 'C'; break;
		case 'A':
		case 'a': Distribution = 'A'; break;
		default: cout << "Unknown data distribution error." << endl; return;
	}

// 	dou_data << Count << " " << Dimensions << endl;
	
	switch (Distribution) {
		case 'E': 
			srand((unsigned)time(NULL));
			GenerateDataEqually(center, Dimensions , Count, 1); 
			break;
		case 'C': 
			srand((unsigned)time(NULL));
			GenerateDataCorrelated(center, Dimensions , Count, 1); 
			break;
		case 'A': 
			srand((unsigned)time(NULL));
			GenerateDataAnticorrelated(center, Dimensions , Count, 1); 
			break;
	}
// 	cout << "done." << endl;
}


//argv[1]: MIN_GROUP_SIZE, argv[2]: MAX_GROUP_SIZE, argv[3]: dimension, argv[4]: groupNo, argv[5]: distribution
int main(int argc, char *argv[])	
{
	int dimension = 5;
	long groupNo = 100000;
	char type='A';
	char path[128]="/Users/wangweicheng/desktop/generator";

#ifdef FOR_RELEASE
	
	if( argc <= 4 )
	{
		cerr << "data generator for pskyline" << endl
				<< "Usage: " << argv[0] << endl
				<< "  dimension: " << endl
				<< "  the number of centers to be generated: " << endl
				<< "  distribution for the centers of groups (E(qually) | C(orrelated) | A(nti-correlated)): " << endl
				<< "  path " << endl;
	}
	else
	{
		dimension = atoi(argv[1]);
		groupNo = atol(argv[2]);
		type = argv[3][0];	
		strcpy( path , argv[4] );
	}

#endif
		
		char filename[128];
	 	sprintf( filename, "%s/%dd_%c_%d.txt" , path , dimension, type, groupNo);

		ofstream out(filename, ios_base::trunc);

		assert(out.is_open());

		out.setf(ios::fixed, ios::floatfield);		
		out << groupNo << " " << dimension << endl;
		
		srand(time(NULL));
		
		double** center;
		center = new double*[groupNo];
		for(long i = 0; i < groupNo; ++i)
		{
			center[i] = new double[dimension];
		}

		GenerateCenter(center, dimension, type, groupNo);
		
		for ( int i=0; i< groupNo; i++ )
		{
			for ( int d=0; d< dimension; d++ )
			out<< center[i][d] * DOMAIN  << "	";

			out << endl;
		}
		
		cout << "generation complete" << endl;

  	for(long i = 0; i < groupNo; ++i)
  	{
  		delete[] center[i];
  	}
  	delete[] center;

  return EXIT_SUCCESS;

}