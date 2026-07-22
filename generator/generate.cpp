// ***************************************************************************
// ** Datengenerierung                                                      **
// ***************************************************************************
//#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#ifdef CYGWIN
	#define MAXINT 2147483647
#else
	#define MAXINT 2147483647
#endif

#define sqr(a) ((a)*(a))


int Statistics_Count;
double* Statistics_SumX;
double* Statistics_SumXsquared;
double* Statistics_SumProduct;


void InitStatistics(int Dimensions)
//   ==============
// initialisiert Z�hlvariablen der Statistik
{
	Statistics_SumX = new double[Dimensions];
	Statistics_SumXsquared = new double[Dimensions];
	Statistics_SumProduct = new double[Dimensions*Dimensions];

	Statistics_Count = 0;
	for (int d=0; d<Dimensions; d++) {
		Statistics_SumX[d]=0.0;
		Statistics_SumXsquared[d]=0.0;
		for (int dd=0; dd<Dimensions; dd++) Statistics_SumProduct[d*Dimensions+dd] = 0.0;
	}
}


void EnterStatistics(int Dimensions,double* x)
//   ===============
// registiriert den Vektor "x" f�r die Statistik
{
	Statistics_Count++;
	for (int d=0; d<Dimensions; d++) {
		Statistics_SumX[d] += x[d];
		Statistics_SumXsquared[d] += x[d]*x[d];
		for (int dd=0; dd<Dimensions; dd++) Statistics_SumProduct[d*Dimensions+dd] += x[d]*x[dd];
	}
}


void OutputStatistics(int Dimensions)
//   ================
// gibt die Statistik aus
{
	int d;
	for (d=0; d<Dimensions; d++) {
		double E = Statistics_SumX[d] / Statistics_Count;
		double V = Statistics_SumXsquared[d]/Statistics_Count - E*E;
		double s = sqrt(V);
		printf("E[X%d]=%5.2f  Var[X%d]=%5.2f  s[X%d]=%5.2f\n",d+1,E,d+1,V,d+1,s);
	}
	printf("\nKorrelationsmatrix:\n");
	for (d=0; d<Dimensions; d++) {
		for (int dd=0; dd<Dimensions; dd++) {
			double Kov = (Statistics_SumProduct[d*Dimensions+dd]/Statistics_Count) -
			             (Statistics_SumX[d]/Statistics_Count) * (Statistics_SumX[dd]/Statistics_Count);
			double Cor = Kov /
			             sqrt(Statistics_SumXsquared[d]/Statistics_Count - sqr(Statistics_SumX[d] / Statistics_Count)) /
			             sqrt(Statistics_SumXsquared[dd]/Statistics_Count - sqr(Statistics_SumX[dd] / Statistics_Count));
			printf("  %5.2f",Cor);
		}
		printf("\n");
	}
	printf("\n");
}


double RandomEqual(double min,double max)
//     ===========
// liefert eine im Intervall [min,max[ gleichverteilte Zufallszahl
{
	double x = (double)rand()/RAND_MAX;
	return x*(max-min)+min;
}


double RandomPeak(double min,double max,int dim)
//     ==========
// liefert eine Zufallsvariable im Intervall [min,max[
// als Summe von "dim" gleichverteilten Zufallszahlen
{
	double sum = 0.0;
	for (int d=0; d<dim; d++) sum += RandomEqual(0,1);
	sum /= dim;
	return sum*(max-min)+min;
}


double RandomNormal(double med,double var)
//     ============
// liefert eine normalverteilte Zufallsvariable mit Erwartungswert med
// im Intervall ]med-var,med+var[
{
	return RandomPeak(med-var,med+var,12);
}


double **GenerateDataEqually(FILE* f,int Count,int Dimensions)
//   ===================
// generiert in der Datei "f" "Count" gleichverteilte Datens�tze
{
	double **returnValue;

	returnValue = (double **) malloc(sizeof(double *)*Count);
	InitStatistics(Dimensions);
	for (int i=0; i<Count; i++)
	{
		//double x[Dimensions];
		double *x;
		x = (double *) malloc(sizeof(double)*Dimensions);
		for (int d=0; d<Dimensions; d++)
		{
			x[d] = RandomEqual(0,1);
			//fprintf(f,"%8.6f ",x[d]);
		}
		EnterStatistics(Dimensions,x);
		//fprintf(f,"\n");

		returnValue[i] = x;
	}
	OutputStatistics(Dimensions);

	return returnValue;
}


double **GenerateDataCorrelated(FILE* f,int Count,int Dimensions)
//   ======================
// generiert in der Datei "f" "Count" korrelierte Datens�tze
{
	double **returnValue;
	int d;
	int i;

	returnValue = (double **) malloc(sizeof(double *)*Count);

	InitStatistics(Dimensions);
	//double x[Dimensions];
	
	for (i=0; i<Count; i++) {
		double *x;
		x = (double *) malloc(sizeof(double)*Dimensions);
again:
		double v = RandomPeak(0,1,Dimensions);
		for (d=0; d<Dimensions; d++) x[d] = v;
		double l = v<=0.5 ? v:1.0-v;
		for (d=0; d<Dimensions; d++) {
			double h = RandomNormal(0,l);
			x[d] += h;
			x[(d+1)%Dimensions] -= h;
		}
		for (d=0; d<Dimensions; d++) if (x[d]<0 || x[d]>=1) goto again;
		//for (d=0; d<Dimensions; d++) fprintf(f,"%8.6f ",x[d]);
		EnterStatistics(Dimensions,x);
		//fprintf(f,"\n");

		returnValue[i] = x;
	}
	OutputStatistics(Dimensions);

	return returnValue;
}


double **GenerateDataAnticorrelated(FILE* f,int Count,int Dimensions)
//   ==========================
// generiert in der Datei "f" "Count" antikorrelierte Datens�tze
{
	double **returnValue;
	double *max;

	returnValue = (double **) malloc(sizeof(double *)*Count);
	max = (double *) malloc(sizeof(double) * Dimensions);

	for(int i = 0; i < Dimensions; i++)
	{
		max[i] = -MAXINT;
	}

	InitStatistics(Dimensions);
	//double x[Dimensions];
	double *x;
	int d;
	int i;
	printf("%d\n", Count);
	for (i=0; i<Count; i++) 
	{
		x = (double *) malloc(sizeof(double)*Dimensions);
again:
		double v = RandomNormal(0.5,0.25);
		for (d=0; d<Dimensions; d++) x[d] = v;
		double l = v<=0.5 ? v:1.0-v;
		for (d=0; d<Dimensions; d++) {
			double h = RandomEqual(-l,l);
			x[d] += h;
			x[(d+1)%Dimensions] -= h;
		}
		printf("point %d \n", i);
		for (d=0; d<Dimensions; d++) if (x[d]<0 || x[d]>=1) goto again;
		//for (d=0; d<Dimensions; d++) fprintf(f,"%8.6f ",x[d]);
		EnterStatistics(Dimensions,x);
		//fprintf(f,"\n");
		returnValue[i] = x;
		for(int j = 0; j < Dimensions; j++)
		{
			if(x[j] > max[j] )
				max[j] = x[j];
		}
	}
	
	for(int i = 0; i < Count; i++)
	{
		for(int j = 0; j < Dimensions; j++)
		{
			returnValue[i][j] /= max[j];
		}
	}

	OutputStatistics(Dimensions);

	return returnValue;
}


double *generateZipProb(double theta, int noOfElt)
{
	double *returnValue;
	int i;

	double total;

	int rank;
	double A;

	returnValue = (double *) malloc(sizeof(double)*noOfElt);
	memset(returnValue, 0, sizeof(double)*noOfElt);

	total = 0;
	for (i = 0; i < noOfElt; i++)
	{
		rank = i+1;
		total += 1.0/pow(rank, theta);
	}

	A = 1.0/total;

	for (i = 0; i < noOfElt; i++)
	{
		rank = i+1;
		returnValue[i] = A/pow(rank, theta);
	}

	return returnValue;
}

int generateRandomClass(int noOfElt, double *prob)
{
	int i;

	double *accum;

	int returnValue;

	double min, max;
	double randNo;

	accum = (double *) malloc(sizeof(double)*noOfElt);
	memset(accum, 0, sizeof(double)*noOfElt);

	accum[0] = prob[0];
	for (i = 1; i < noOfElt; i++)
	{
		accum[i] = accum[i-1]+prob[i];
	}

	min = 0;
	max = accum[noOfElt-1];

	randNo = RandomEqual(min, max);

	returnValue = -1;
	for (i = 0; i < noOfElt; i++)
	{
		if (randNo <= accum[i])
		{
			returnValue = i;
			break;
		}
	}

	free(accum);

	return returnValue;
}

//catData = GenerateCatData(Count, noOfCatDim, noOfClass);
int **GenerateCatData(int noOfData, int noOfCatDim, int noOfClass, double theta)
{
	int **returnValue;
	int i, j;

	double *prob;

	int classNo;
	
	if (noOfCatDim > 0)
	{
		returnValue = (int **) malloc(sizeof(int *)*noOfData);

		for (i = 0; i < noOfData; i++)
		{
			returnValue[i] = (int *) malloc(sizeof(int)*noOfCatDim);
		}

		// generate prob.
		prob = generateZipProb(theta, noOfClass);

		for (i = 0; i < noOfData; i++)
		{
			for (j = 0; j < noOfCatDim; j++)
			{
				classNo = generateRandomClass(noOfClass, prob);
				returnValue[i][j] = classNo;
			}
		}
		
		free(prob);
	}
	else
	{
		returnValue = NULL;
	}

	return returnValue;
}

void OutputData(FILE *filePtr, int noOfData, double **numData, int **catData, int noOfNumDim, int noOfCatDim)
{
	int i, j;

	double aNo;

	double scale;

	scale = 1;

	for (i = 0; i < noOfData; i++)
	{
		for (j = 0; j < noOfNumDim; j++)
		{
			/*
			if (numData[i][j] == 1.0)
			{
				aNo = scale-1;
			}
			else
			{
				aNo = (int) (numData[i][j]*scale);
			}
			*/
			fprintf(filePtr, "%lf", numData[i][j]);
			if (j != noOfNumDim -1)
			{
				fprintf(filePtr, " ");
			}
		}

		if (noOfCatDim > 0)
		{
			fprintf(filePtr, " ");
		}

		for (j = 0; j < noOfCatDim; j++)
		{
			fprintf(filePtr, "%d", catData[i][j]);
			if (j != noOfNumDim - 1)
			{
				fprintf(filePtr, " ");
			}
		}

		fprintf(filePtr, "\n");
	}
}

void GenerateData(int noOfNumDim, int noOfCatDim, int noOfClass, double theta, char Distribution, int Count,char* FileName)
//   ============
// generierte eine Datei mit zuf�lligen Daten
{
	double **numData;
	int **catData;
	if (Count <= 0) {
		//printf("Ung�ltige Anzahl von Punkten.\n");
		printf("No. of tuples > 0.\n");
		return;
	}
	if (noOfNumDim < 2) {
		//printf("Ung�ltige Anzahl von Dimensionen.\n");
		printf("No. of Num Dims should be >= 2.\n");
		return;
	}
	if (noOfCatDim < 0)
	{
		printf("No. of Cat Dims should be >= 0.\n");
		return;
	}
	if (noOfClass < 1)
	{
		printf("No. of Classes should be >= 1.\n");
		return;
	}
	switch (Distribution) {
	case 'E':
	case 'e': Distribution = 'E'; break;
	case 'C':
	case 'c': Distribution = 'C'; break;
	case 'A':
	case 'a': Distribution = 'A'; break;
	//default: printf("Ung�ltige Verteilung.\n"); return;
		default: printf("Invalid distribution.\n"); return;
	}

	FILE* f = fopen(FileName,"wt");
	if (f == NULL) {
		//printf("Kann Datei \"%s\" nicht anlegen.\n",FileName);
		printf("File \"%s\" cannot be opened.\n",FileName);
		return;
	}
	fprintf(f,"%d %d\n",Count,noOfNumDim);
	switch (Distribution) {
	case 'E': 
		numData = GenerateDataEqually(f,Count,noOfNumDim); 
		break;
	case 'C': 
		numData = GenerateDataCorrelated(f,Count,noOfNumDim); 
		break;
	case 'A': 
		numData = GenerateDataAnticorrelated(f,Count,noOfNumDim); 
		break;
	}
	catData = GenerateCatData(Count, noOfCatDim, noOfClass, theta);

	OutputData(f, Count, numData, catData, noOfNumDim, noOfCatDim);

	fclose(f);
	//printf("%d Punkte generiert, Datei \"%s\".\n",Count,FileName);
	printf("%d data points were generated in data \"%s\".\n",Count,FileName);
}



int main(int argc, char** argv)
//  ====
// main program
{
	for(int i = 0; i < 10; i++)
		rand();

	if (argc != 8) {
		//printf("generate - Generierung von Testdaten f�r Skyline\n\n");
		//printf("Syntax: generate <Dimensionen> <Verteilung> <Anzahl> <Datei>\n");
		//printf("        mit Verteilung = E(qually) | C(orrelated) | A(nti-correlated)\n\n");
		printf("generate - Generatintg data Set for Skyline\n\n");
		printf("Syntax: generate <No. Of Num Dims> <No. of Categorical Dims> <No. of Classes> <Zipfian Theta Parameter> <Distribution> <No. Of Tuples> <Filename>\n");
		printf("        where Distribution = E(qually) | C(orrelated) | A(nti-correlated)\n\n");
	} else {
		//for(int i = 0; i < 2; i++)
		//	rand();

		int noOfNumDim = atoi(argv[1]);
		int noOfCatDim = atoi(argv[2]);
		int noOfClass = atoi(argv[3]);
		double theta = (double) atof(argv[4]);
		char Distribution = argv[5][0];
		int Count = atoi(argv[6]);
		char *Filename = argv[7];
		GenerateData(noOfNumDim, noOfCatDim, noOfClass, theta, Distribution, Count, Filename);
	}
	

	/*
	FILE* c_fp;
	
	if( ( c_fp = fopen( "temp.txt", "r")) == NULL)
	{
		fprintf( stderr, "Cannot open the config file.\n");
		exit(0);
	}

	int d = 4;
	int n = 1372;

	double* max = new double[d];
	double* min = new double[d];

	for(int i = 0; i < d; i++)
	{
		max[i] = -10000;
		min[i] = 10000;
	}

	double next;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < d; j++)
		{
			fscanf( c_fp, "%lf", &next);
			if(next > max[j])
				max[j] = next;
			if(next < min[j])
				min[j] = next;
		}
	}
	fclose(c_fp);


	c_fp = fopen( "temp.txt", "r");
	FILE* w_fp;
	w_fp = fopen( "output.txt", "w");

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < d; j++)
		{
			fscanf( c_fp, "%lf", &next);
			next = (next - min[j]) / (max[j] - min[j]);
			fprintf(w_fp, "%lf ", next);
		}
		fprintf(w_fp, "\n");
	}
	fclose(c_fp);
	fclose(w_fp);

	*/
	return 0;
}

