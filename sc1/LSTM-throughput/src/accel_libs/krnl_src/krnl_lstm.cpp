#include <hls_math.h>
#include <ap_fixed.h>
#include <ap_int.h>

#define KRNLS 4
#define MAX_BATCH 256

#define TIMESTEPS 64
#define DATA_DIM 1
#define UNITS1 128
#define UNITS2 128
#define MAX_UNITS 128  //max between UNITS1, UNITS2
#define MAX_MCOL 256 //UNITS1+UNITS2  //max between UNITS1+DATA_DIM, UNITS2+UNITS1

const int  LVL_SEQ[2] {1,0};

const int steps = TIMESTEPS;
const int dim = DATA_DIM;
const int sdim = KRNLS*TIMESTEPS*DATA_DIM; //max between DATA_DIM
const int sdim2 = MAX_BATCH*KRNLS*TIMESTEPS*DATA_DIM; //max between DATA_DIM
const int un1 = UNITS1;
const int un2 = UNITS2;
const int min_un = UNITS1; //min between UNITS1, UNITS2
const int max_un = UNITS2; //max between UNITS1, UNITS2
const int max_mtx = MAX_UNITS*MAX_MCOL; //
const int max_vct = 2*MAX_UNITS; //
const int max_dmtx = MAX_UNITS*DATA_DIM; //
const int max_dvct = DATA_DIM; //
const int min_mcol = UNITS1+DATA_DIM; //min between UNITS1+DATA_DIM, UNITS1+UNITS2
const int max_mcol = UNITS1+UNITS2; //max between UNITS1+DATA_DIM, UNITS1+UNITS2

#define FRAQ 10 //8
#define FRAQ_CONV 1024 //256 //2^8
#define FRAQ2 12 //8
#define FRAQ2_CONV 4096 //256 //2^8
typedef ap_fixed<16, 6, AP_RND>  dtype;
typedef ap_fixed<18, 6, AP_RND>  dtype2;

using namespace std;

dtype sigmoid(dtype2 x)
{
	#pragma HLS PIPELINE
	float input = x.to_float();
	float exp_mval = exp(-input);
	float divident = ((float)1) + exp_mval;
	float result_tmp = ((float)1)/divident;
	dtype result = result_tmp;
	return result;
}

void array_sigmoid(dtype2 x[MAX_UNITS], dtype y[MAX_UNITS])
{
	LOOP_SIGMOID:for(int i=0; i<MAX_UNITS; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		//#pragma HLS PIPELINE
		y[i] = sigmoid(x[i]);
	}
}

dtype tanh_fx(dtype2 x)
{
	#pragma HLS PIPELINE
	float input = x.to_float();
	float result_tmp = tanh(input);
	dtype result = result_tmp;
	return result;
}

void array_tanh(dtype2 x[MAX_UNITS], dtype y[MAX_UNITS])
{
	LOOP_ITANH:for(int i=0; i<MAX_UNITS; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		//#pragma HLS PIPELINE
		y[i] = tanh_fx(x[i]);
	}
}

dtype2 row_vector_mul(const dtype w[MAX_MCOL], dtype h_x[MAX_MCOL], const dtype b)
{
	dtype2 res;
	dtype2 first;
	dtype2 temp;
	LOOP_INTERNAL_MATRIX:for(int j=0; j<MAX_MCOL; j++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_mcol max = max_mcol
		first = (j==0) ? (dtype2)b : res;
		temp = (dtype2)w[j] * (dtype2)h_x[j];
		res = first + temp;
	}
	return res;
}

void input_gate(const dtype Wi[MAX_UNITS][MAX_MCOL], const dtype bi[MAX_UNITS], dtype h_xi[MAX_MCOL], dtype out_i[MAX_UNITS])
{
	#pragma HLS ARRAY_PARTITION variable=Wi dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bi complete
	#pragma HLS ARRAY_PARTITION variable=h_xi complete
	dtype2 i_t[MAX_UNITS];
	LOOP_ROW_I:for(int i=0; i<MAX_UNITS; ++i) 
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		i_t[i] = row_vector_mul(Wi[i], h_xi, bi[i]);
	}
	array_sigmoid(i_t, out_i);
}

void forget_gate(const dtype Wf[MAX_UNITS][MAX_MCOL], const dtype bf[MAX_UNITS], dtype h_xf[MAX_MCOL], dtype out_f[MAX_UNITS])
{
	#pragma HLS ARRAY_PARTITION variable=Wf dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bf complete
	#pragma HLS ARRAY_PARTITION variable=h_xf complete
	dtype2 f_t[MAX_UNITS];
	LOOP_ROW_F:for(int i=0; i<MAX_UNITS; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		f_t[i] = row_vector_mul(Wf[i], h_xf, bf[i]);
	}
	array_sigmoid(f_t, out_f);
}

void newcell_gate(const dtype Wc[MAX_UNITS][MAX_MCOL], const dtype bc[MAX_UNITS], dtype h_xc[MAX_MCOL], dtype out_c[MAX_UNITS])
{
	#pragma HLS ARRAY_PARTITION variable=Wc dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bc complete
	#pragma HLS ARRAY_PARTITION variable=h_xc complete
	dtype2 c_t[MAX_UNITS];
	LOOP_ROW_C:for(int i=0; i<MAX_UNITS; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		c_t[i] = row_vector_mul(Wc[i], h_xc, bc[i]);
	}
	array_tanh(c_t, out_c);
}

void output_gate(const dtype Wo[MAX_UNITS][MAX_MCOL], const dtype bo[MAX_UNITS], dtype h_xo[MAX_MCOL], dtype out_o[MAX_UNITS])
{
	#pragma HLS ARRAY_PARTITION variable=Wo dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bo complete
	#pragma HLS ARRAY_PARTITION variable=h_xo complete
	dtype2 o_t[MAX_UNITS];
	LOOP_ROW_O:for(int i=0; i<MAX_UNITS; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		o_t[i] = row_vector_mul(Wo[i], h_xo, bo[i]);
	}
	array_sigmoid(o_t, out_o);
}

dtype2 dense_vector_mul(const dtype w[MAX_UNITS], dtype h_x[MAX_UNITS], const dtype b)
{
	dtype2 res;
	dtype2 first;
	dtype2 temp;
	LOOP_INTERNAL_MATRIX:for(int j=0; j<MAX_UNITS; j++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		first = (j==0) ? (dtype2)b : res;
		temp = (dtype2)w[j] * (dtype2)h_x[j];
		res = first + temp;
	}
	return res;
}

void Dense(const dtype W[DATA_DIM][MAX_UNITS], const dtype b[DATA_DIM], dtype data[MAX_UNITS], dtype2 res[DATA_DIM])
{
	#pragma HLS ARRAY_PARTITION variable=W dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=b complete
	#pragma HLS ARRAY_PARTITION variable=data complete
	#pragma HLS ARRAY_PARTITION variable=res complete
	LOOP_DIMS:for(int i=0; i<DATA_DIM; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = dim max = dim
		#pragma HLS PIPELINE II = 1
		res[i]=dense_vector_mul(W[i], data, b[i]);
	}
}

void lstm_core(const dtype Wi[MAX_UNITS][MAX_MCOL], const dtype bi[MAX_UNITS], const dtype Wf[MAX_UNITS][MAX_MCOL], const dtype bf[MAX_UNITS], const dtype Wc[MAX_UNITS][MAX_MCOL], const dtype bc[MAX_UNITS], const dtype Wo[MAX_UNITS][MAX_MCOL], const dtype bo[MAX_UNITS], const dtype W[DATA_DIM][MAX_UNITS], const dtype b[DATA_DIM], dtype input[TIMESTEPS][MAX_UNITS], dtype2 output[TIMESTEPS][DATA_DIM], int tsteps, int layer)
{
	#pragma HLS ARRAY_PARTITION variable=Wi dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bi complete
	#pragma HLS ARRAY_PARTITION variable=Wf dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bf complete
	#pragma HLS ARRAY_PARTITION variable=Wc dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bc complete
	#pragma HLS ARRAY_PARTITION variable=Wo dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=bo complete
	#pragma HLS ARRAY_PARTITION variable=W dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=b complete
	#pragma HLS ARRAY_PARTITION variable=input dim=1
	#pragma HLS ARRAY_PARTITION variable=output dim=1
	dtype c[MAX_UNITS], h[MAX_UNITS];
	dtype c_f[MAX_UNITS], c_s[MAX_UNITS], c_tmp[MAX_UNITS], c_tanh_tmp[MAX_UNITS], h_tmp[MAX_UNITS];
	dtype x_h[MAX_MCOL];
	dtype f_t[MAX_UNITS], i_t[MAX_UNITS], c_t[MAX_UNITS], o_t[MAX_UNITS];
	dtype f_t2[MAX_UNITS], i_t2[MAX_UNITS], c_t2[MAX_UNITS], o_t2[MAX_UNITS];
	int pos=0;
	LOOP_TIMESTEPS:for(int k=0; k<tsteps; ++k)
	{
		#pragma HLS LOOP_TRIPCOUNT min = 1 max = steps
		pos = (LVL_SEQ[layer]==0) ? (tsteps-1) : k;
		LOOP_INPUT:for(int i=0; i<MAX_MCOL; ++i)
		{
			#pragma HLS LOOP_TRIPCOUNT min = min_mcol max = max_mcol
			if(layer==0)
				x_h[i] = (i<DATA_DIM) ? input[pos][i] : ((i-DATA_DIM)<MAX_UNITS && k>0) ? h[i-DATA_DIM] : ((dtype) 0.0);
			else if(layer==1)
				x_h[i] = (i<UNITS1) ? input[pos][i] : ((i-UNITS1)<MAX_UNITS && k>0) ? h[i-UNITS1] : ((dtype) 0.0);
		}

		input_gate(Wi, bi, x_h, i_t2);
		forget_gate(Wf, bf, x_h, f_t2);
		newcell_gate(Wc, bc, x_h, c_t2);
		output_gate(Wo, bo, x_h, o_t2);

		LOOP_IN_UNITS:for(int i=0; i<MAX_UNITS; ++i)
		{
			#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
			#pragma HLS PIPELINE II = 1
			c_f[i] = (k>0) ? ((dtype)(f_t2[i]*c[i])) : ((dtype) 0.0);
			c_s[i] = c_t2[i]*i_t2[i];
			c_tmp[i] = c_f[i] + c_s[i];
			c_tanh_tmp[i] = tanh_fx((dtype2)c_tmp[i]);
			h_tmp[i] = o_t2[i] * c_tanh_tmp[i];
			c[i] = c_tmp[i];
			h[i] = h_tmp[i];
			input[k][i] = h_tmp[i];
		}
		if(layer==1)
		{
			Dense(W, b, h_tmp, output[k]);
		}
	}
}

extern "C" {
void krnl_lstm(ap_int<512> *data, ap_int<512> *iWi, ap_int<512> *ibi, ap_int<512> *iWf, ap_int<512> *ibf, ap_int<512> *iWc, ap_int<512> *ibc, ap_int<512> *iWo, ap_int<512> *ibo, ap_int<512> *iW, ap_int<16> *ib, short int *res, int active_timesteps, int batch)
{
	#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=iWi offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=ibi offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=iWf offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=ibf offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=iWc offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=ibc offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=iWo offset=slave bundle=gmem3
	#pragma HLS INTERFACE m_axi port=ibo offset=slave bundle=gmem3
	#pragma HLS INTERFACE m_axi port=iW offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=ib offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=res offset=slave bundle=gmem0
	#pragma HLS INTERFACE s_axilite port=data bundle=control
	#pragma HLS INTERFACE s_axilite port=iWi bundle=control
	#pragma HLS INTERFACE s_axilite port=ibi bundle=control
	#pragma HLS INTERFACE s_axilite port=iWf bundle=control
	#pragma HLS INTERFACE s_axilite port=ibf bundle=control
	#pragma HLS INTERFACE s_axilite port=iWc bundle=control
	#pragma HLS INTERFACE s_axilite port=ibc bundle=control
	#pragma HLS INTERFACE s_axilite port=iWo bundle=control
	#pragma HLS INTERFACE s_axilite port=ibo bundle=control
	#pragma HLS INTERFACE s_axilite port=iW bundle=control
	#pragma HLS INTERFACE s_axilite port=ib bundle=control
	#pragma HLS INTERFACE s_axilite port=res bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control


	int actv_timesteps = active_timesteps;
	int batch_size = batch;

	dtype input_k1[TIMESTEPS][MAX_UNITS];
	dtype input_k2[TIMESTEPS][MAX_UNITS];
	dtype input_k3[TIMESTEPS][MAX_UNITS];
	dtype input_k4[TIMESTEPS][MAX_UNITS];
	#pragma HLS ARRAY_PARTITION variable=input_k1 dim=1
	#pragma HLS ARRAY_PARTITION variable=input_k2 dim=1
	#pragma HLS ARRAY_PARTITION variable=input_k3 dim=1
	#pragma HLS ARRAY_PARTITION variable=input_k4 dim=1
	dtype Wik1_reg[UNITS1][UNITS1+UNITS2];
	dtype Wik2_reg[UNITS1][UNITS1+UNITS2];
	dtype Wik3_reg[UNITS1][UNITS1+UNITS2];
	dtype Wik4_reg[UNITS1][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wik1_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wik2_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wik3_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wik4_reg dim=2 complete
	dtype bik1_reg[UNITS1];
	dtype bik2_reg[UNITS1];
	dtype bik3_reg[UNITS1];
	dtype bik4_reg[UNITS1];
	#pragma HLS ARRAY_PARTITION variable=bik1_reg complete
	#pragma HLS ARRAY_PARTITION variable=bik2_reg complete
	#pragma HLS ARRAY_PARTITION variable=bik3_reg complete
	#pragma HLS ARRAY_PARTITION variable=bik4_reg complete
	dtype Wfk1_reg[UNITS1][UNITS1+UNITS2];
	dtype Wfk2_reg[UNITS1][UNITS1+UNITS2];
	dtype Wfk3_reg[UNITS1][UNITS1+UNITS2];
	dtype Wfk4_reg[UNITS1][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wfk1_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wfk2_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wfk3_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wfk4_reg dim=2 complete
	dtype bfk1_reg[UNITS1];
	dtype bfk2_reg[UNITS1];
	dtype bfk3_reg[UNITS1];
	dtype bfk4_reg[UNITS1];
	#pragma HLS ARRAY_PARTITION variable=bfk1_reg complete
	#pragma HLS ARRAY_PARTITION variable=bfk2_reg complete
	#pragma HLS ARRAY_PARTITION variable=bfk3_reg complete
	#pragma HLS ARRAY_PARTITION variable=bfk4_reg complete
	dtype Wck1_reg[UNITS1][UNITS1+UNITS2];
	dtype Wck2_reg[UNITS1][UNITS1+UNITS2];
	dtype Wck3_reg[UNITS1][UNITS1+UNITS2];
	dtype Wck4_reg[UNITS1][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wck1_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wck2_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wck3_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wck4_reg dim=2 complete
	dtype bck1_reg[UNITS1];
	dtype bck2_reg[UNITS1];
	dtype bck3_reg[UNITS1];
	dtype bck4_reg[UNITS1];
	#pragma HLS ARRAY_PARTITION variable=bck1_reg complete
	#pragma HLS ARRAY_PARTITION variable=bck2_reg complete
	#pragma HLS ARRAY_PARTITION variable=bck3_reg complete
	#pragma HLS ARRAY_PARTITION variable=bck4_reg complete
	dtype Wok1_reg[UNITS1][UNITS1+UNITS2];
	dtype Wok2_reg[UNITS1][UNITS1+UNITS2];
	dtype Wok3_reg[UNITS1][UNITS1+UNITS2];
	dtype Wok4_reg[UNITS1][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wok1_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wok2_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wok3_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wok4_reg dim=2 complete
	dtype bok1_reg[UNITS1];
	dtype bok2_reg[UNITS1];
	dtype bok3_reg[UNITS1];
	dtype bok4_reg[UNITS1];
	#pragma HLS ARRAY_PARTITION variable=bok1_reg complete
	#pragma HLS ARRAY_PARTITION variable=bok2_reg complete
	#pragma HLS ARRAY_PARTITION variable=bok3_reg complete
	#pragma HLS ARRAY_PARTITION variable=bok4_reg complete
	dtype Wk1_reg[DATA_DIM][UNITS2];
	dtype Wk2_reg[DATA_DIM][UNITS2];
	dtype Wk3_reg[DATA_DIM][UNITS2];
	dtype Wk4_reg[DATA_DIM][UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wk1_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wk2_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wk3_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wk4_reg dim=2 complete
	dtype bk1_reg[DATA_DIM];
	dtype bk2_reg[DATA_DIM];
	dtype bk3_reg[DATA_DIM];
	dtype bk4_reg[DATA_DIM];
	#pragma HLS ARRAY_PARTITION variable=bk1_reg complete
	#pragma HLS ARRAY_PARTITION variable=bk2_reg complete
	#pragma HLS ARRAY_PARTITION variable=bk3_reg complete
	#pragma HLS ARRAY_PARTITION variable=bk4_reg complete
	dtype2 output_k1[TIMESTEPS][DATA_DIM];
	dtype2 output_k2[TIMESTEPS][DATA_DIM];
	dtype2 output_k3[TIMESTEPS][DATA_DIM];
	dtype2 output_k4[TIMESTEPS][DATA_DIM];
	#pragma HLS ARRAY_PARTITION variable=output_k1 dim=1
	#pragma HLS ARRAY_PARTITION variable=output_k2 dim=1
	#pragma HLS ARRAY_PARTITION variable=output_k3 dim=1
	#pragma HLS ARRAY_PARTITION variable=output_k4 dim=1
	
	//Considering 64 timesteps (max) of 16bit values then each input requires 2-512bit packets.
        //So for MAX_BATCH input size we require 2x 512 bit storage.
	ap_int<512> Din[2*MAX_BATCH];

	short int outbatch[MAX_BATCH*TIMESTEPS*DATA_DIM];
 
	for(int i=0; i< 2*batch_size; i++)
	{
		Din[i]=data[i];
	}

	for(int i = 0, x=0, y=0; i< ((DATA_DIM*UNITS2*16)/512); i++, y++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
		#pragma HLS PIPELINE II = 1
		ap_int<512> W = iW[i];
		if (y == ((UNITS2*16)/512)){
			x++;
			y = 0;
		}
		for(int j=0;j<(512/32);j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
			#pragma HLS PIPELINE II = 1
			short int W_tmp;
			for(int z=0;z<2;z++)
			{
				W_tmp = (z==0) ? W.range(((j+1)*32)-17, j*32) : W.range(((j+1)*32)-1, (j*32)+16);
				Wk1_reg[x][y*(512/16)+j*2+z]=((float)W_tmp)/FRAQ_CONV;
				Wk2_reg[x][y*(512/16)+j*2+z]=((float)W_tmp)/FRAQ_CONV;
				Wk3_reg[x][y*(512/16)+j*2+z]=((float)W_tmp)/FRAQ_CONV;
				Wk4_reg[x][y*(512/16)+j*2+z]=((float)W_tmp)/FRAQ_CONV;
			}
		}
	}
	for(int i = 0, x = 0 ; i< DATA_DIM; i++, x++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = max_dvct max = max_dvct
		#pragma HLS PIPELINE II = 1
		bk1_reg[x] = ((float)ib[i])/FRAQ_CONV;
		bk2_reg[x] = ((float)ib[i])/FRAQ_CONV;
		bk3_reg[x] = ((float)ib[i])/FRAQ_CONV;
		bk4_reg[x] = ((float)ib[i])/FRAQ_CONV;
	}

	LOOP_BATCH:for(int bh=0; bh<(2*batch_size)/(KRNLS*((TIMESTEPS*DATA_DIM)/32)); ++bh)
	{
		#pragma HLS LOOP_TRIPCOUNT min = 32 max = 32
		for(int i = 0, x=0, y=0; i< KRNLS*((TIMESTEPS*DATA_DIM)/32); i++, y++)
		{
			#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
			#pragma HLS PIPELINE II = 1
			ap_int<512> Dt = Din[i+bh*KRNLS*((TIMESTEPS*DATA_DIM)/32)];
			if (y == DATA_DIM) //Because y dim is smaller else it should have been /32
			{
				if(x == TIMESTEPS-1)
					x = 0;
				else
					x++;
				y = 0;
			}
			for(int j=0;j<(512/32);j++)
			{
				#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
				#pragma HLS PIPELINE II = 1
				short int d_tmp;
				for(int z=0;z<2;z++)
				{
					d_tmp = (z==0) ? Dt.range(((j+1)*32)-17, j*32) : Dt.range(((j+1)*32)-1, (j*32)+16);
					if(i<((TIMESTEPS*DATA_DIM)/32))				
						input_k1[x*(512/16)+j*2+z][y]=((float)d_tmp)/FRAQ_CONV;
					else if(i>=((TIMESTEPS*DATA_DIM)/32) && i<2*((TIMESTEPS*DATA_DIM)/32))
						input_k2[x*(512/16)+j*2+z][y]=((float)d_tmp)/FRAQ_CONV;
					else if(i>=2*((TIMESTEPS*DATA_DIM)/32) && i<3*((TIMESTEPS*DATA_DIM)/32))
						input_k3[x*(512/16)+j*2+z][y]=((float)d_tmp)/FRAQ_CONV;
					else
						input_k4[x*(512/16)+j*2+z][y]=((float)d_tmp)/FRAQ_CONV;
				}
			}
		}

		
		LOOP_CELLS:for(int cl=0; cl<2; ++cl)
		{
			#pragma HLS LOOP_TRIPCOUNT min = 2 max = 2
			for(int i=0, x=0, y=0; i< ((MAX_UNITS*MAX_MCOL*16)/512); i++, y++)
			{
				#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 1024
				#pragma HLS PIPELINE II = 1
				ap_int<512> Wi = iWi[i+(cl*((MAX_UNITS*MAX_MCOL*16)/512))];
				ap_int<512> Wf = iWf[i+(cl*((MAX_UNITS*MAX_MCOL*16)/512))];
				ap_int<512> Wc = iWc[i+(cl*((MAX_UNITS*MAX_MCOL*16)/512))];
				ap_int<512> Wo = iWo[i+(cl*((MAX_UNITS*MAX_MCOL*16)/512))];

				if (y == ((MAX_MCOL*16)/512))
				{
					x++;
					y = 0;
				}
				for(int j=0;j<(512/32);j++)
				{
					#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
					#pragma HLS PIPELINE II = 1
					short int Wi_tmp;
					short int Wf_tmp;
					short int Wc_tmp;
					short int Wo_tmp;
					for(int z=0;z<2;z++)
					{
						Wi_tmp = (z==0) ? Wi.range(((j+1)*32)-17, j*32) : Wi.range(((j+1)*32)-1, (j*32)+16);
						Wf_tmp = (z==0) ? Wf.range(((j+1)*32)-17, j*32) : Wf.range(((j+1)*32)-1, (j*32)+16);
						Wc_tmp = (z==0) ? Wc.range(((j+1)*32)-17, j*32) : Wc.range(((j+1)*32)-1, (j*32)+16);
						Wo_tmp = (z==0) ? Wo.range(((j+1)*32)-17, j*32) : Wo.range(((j+1)*32)-1, (j*32)+16);
						Wik1_reg[x][y*(512/16)+j*2+z]=((float)Wi_tmp)/FRAQ_CONV;
						Wik2_reg[x][y*(512/16)+j*2+z]=((float)Wi_tmp)/FRAQ_CONV;
						Wik3_reg[x][y*(512/16)+j*2+z]=((float)Wi_tmp)/FRAQ_CONV;
						Wik4_reg[x][y*(512/16)+j*2+z]=((float)Wi_tmp)/FRAQ_CONV;
						Wfk1_reg[x][y*(512/16)+j*2+z]=((float)Wf_tmp)/FRAQ_CONV;
						Wfk2_reg[x][y*(512/16)+j*2+z]=((float)Wf_tmp)/FRAQ_CONV;
						Wfk3_reg[x][y*(512/16)+j*2+z]=((float)Wf_tmp)/FRAQ_CONV;
						Wfk4_reg[x][y*(512/16)+j*2+z]=((float)Wf_tmp)/FRAQ_CONV;
						Wck1_reg[x][y*(512/16)+j*2+z]=((float)Wc_tmp)/FRAQ_CONV;
						Wck2_reg[x][y*(512/16)+j*2+z]=((float)Wc_tmp)/FRAQ_CONV;
						Wck3_reg[x][y*(512/16)+j*2+z]=((float)Wc_tmp)/FRAQ_CONV;
						Wck4_reg[x][y*(512/16)+j*2+z]=((float)Wc_tmp)/FRAQ_CONV;
						Wok1_reg[x][y*(512/16)+j*2+z]=((float)Wo_tmp)/FRAQ_CONV;
						Wok2_reg[x][y*(512/16)+j*2+z]=((float)Wo_tmp)/FRAQ_CONV;
						Wok3_reg[x][y*(512/16)+j*2+z]=((float)Wo_tmp)/FRAQ_CONV;
						Wok4_reg[x][y*(512/16)+j*2+z]=((float)Wo_tmp)/FRAQ_CONV;
					}
				}
			}
			for(int i = 0, x = 0 ; i< ((MAX_UNITS*16)/512); i++, x++)
			{
				#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4
				#pragma HLS PIPELINE II = 1
				ap_int<512> bi = ibi[i+(cl*((MAX_UNITS*16)/512))];
				ap_int<512> bf = ibf[i+(cl*((MAX_UNITS*16)/512))];
				ap_int<512> bc = ibc[i+(cl*((MAX_UNITS*16)/512))];
				ap_int<512> bo = ibo[i+(cl*((MAX_UNITS*16)/512))];
				for(int j=0;j<(512/32);j++)
				{
					#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
					#pragma HLS PIPELINE II = 1
					short int bi_tmp;
					short int bf_tmp;
					short int bc_tmp;
					short int bo_tmp;
					for(int z=0;z<2;z++)
					{
						bi_tmp = (z==0) ? bi.range(((j+1)*32)-17, j*32) : bi.range(((j+1)*32)-1, (j*32)+16);
						bf_tmp = (z==0) ? bf.range(((j+1)*32)-17, j*32) : bf.range(((j+1)*32)-1, (j*32)+16);
						bc_tmp = (z==0) ? bc.range(((j+1)*32)-17, j*32) : bc.range(((j+1)*32)-1, (j*32)+16);
						bo_tmp = (z==0) ? bo.range(((j+1)*32)-17, j*32) : bo.range(((j+1)*32)-1, (j*32)+16);
						bik1_reg[x*(512/16)+j*2+z]=((float)bi_tmp)/FRAQ_CONV;
						bik2_reg[x*(512/16)+j*2+z]=((float)bi_tmp)/FRAQ_CONV;
						bik3_reg[x*(512/16)+j*2+z]=((float)bi_tmp)/FRAQ_CONV;
						bik4_reg[x*(512/16)+j*2+z]=((float)bi_tmp)/FRAQ_CONV;
						bfk1_reg[x*(512/16)+j*2+z]=((float)bf_tmp)/FRAQ_CONV;
						bfk2_reg[x*(512/16)+j*2+z]=((float)bf_tmp)/FRAQ_CONV;
						bfk3_reg[x*(512/16)+j*2+z]=((float)bf_tmp)/FRAQ_CONV;
						bfk4_reg[x*(512/16)+j*2+z]=((float)bf_tmp)/FRAQ_CONV;
						bck1_reg[x*(512/16)+j*2+z]=((float)bc_tmp)/FRAQ_CONV;
						bck2_reg[x*(512/16)+j*2+z]=((float)bc_tmp)/FRAQ_CONV;
						bck3_reg[x*(512/16)+j*2+z]=((float)bc_tmp)/FRAQ_CONV;
						bck4_reg[x*(512/16)+j*2+z]=((float)bc_tmp)/FRAQ_CONV;
						bok1_reg[x*(512/16)+j*2+z]=((float)bo_tmp)/FRAQ_CONV;
						bok2_reg[x*(512/16)+j*2+z]=((float)bo_tmp)/FRAQ_CONV;
						bok3_reg[x*(512/16)+j*2+z]=((float)bo_tmp)/FRAQ_CONV;
						bok4_reg[x*(512/16)+j*2+z]=((float)bo_tmp)/FRAQ_CONV;
					}
				}
			}
			lstm_core(Wik1_reg, bik1_reg, Wfk1_reg, bfk1_reg, Wck1_reg, bck1_reg, Wok1_reg, bok1_reg, Wk1_reg, bk1_reg, input_k1, output_k1, actv_timesteps, cl);
			lstm_core(Wik2_reg, bik2_reg, Wfk2_reg, bfk2_reg, Wck2_reg, bck2_reg, Wok2_reg, bok2_reg, Wk2_reg, bk2_reg, input_k2, output_k2, actv_timesteps, cl);
			lstm_core(Wik3_reg, bik3_reg, Wfk3_reg, bfk3_reg, Wck3_reg, bck3_reg, Wok3_reg, bok3_reg, Wk3_reg, bk3_reg, input_k3, output_k3, actv_timesteps, cl);
			lstm_core(Wik4_reg, bik4_reg, Wfk4_reg, bfk4_reg, Wck4_reg, bck4_reg, Wok4_reg, bok4_reg, Wk4_reg, bk4_reg, input_k4, output_k4, actv_timesteps, cl);
		}
		//for(int i = 0, x=0, y=0; i< KRNLS*actv_timesteps*DATA_DIM; i++, y++)
		for(int i = 0, x=0, y=0; i< KRNLS*TIMESTEPS*DATA_DIM; i++, y++)
		{
			#pragma HLS LOOP_TRIPCOUNT min = sdim max = sdim
			#pragma HLS PIPELINE II = 1
			if (y == DATA_DIM)
			{
				if(x == TIMESTEPS-1)
					x = 0;
				else
					x++;
				y = 0;
			}
			outbatch[i+bh*KRNLS*TIMESTEPS*DATA_DIM] = (i<TIMESTEPS*DATA_DIM) ? output_k1[x][y]*FRAQ2_CONV : (i>=TIMESTEPS*DATA_DIM && i<2*TIMESTEPS*DATA_DIM) ? output_k2[x][y]*FRAQ2_CONV : 
				 (i>=2*TIMESTEPS*DATA_DIM && i<3*TIMESTEPS*DATA_DIM) ? output_k3[x][y]*FRAQ2_CONV : output_k4[x][y]*FRAQ2_CONV;
		}
	}
	for(int i=0; i< batch_size*TIMESTEPS*DATA_DIM; i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = sdim2 max = sdim2
		#pragma HLS PIPELINE II = 1
		res[i]=outbatch[i];
	}
}
}
