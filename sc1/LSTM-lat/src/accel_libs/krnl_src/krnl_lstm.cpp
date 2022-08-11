//---------------------------------//
//- Author: Stamoulias Ioannis    -//
//- email: jstamoulias@gmail.com  -//
//---------------------------------//

#include <hls_math.h>
#include <ap_fixed.h>
#include <ap_int.h>

#define TIMESTEPS 64
#define DATA_DIM 1
#define UNITS1 128
#define UNITS2 128
#define MAX_UNITS 128  //max between UNITS1, UNITS2
#define MAX_MCOL 256 //UNITS1+UNITS2  //max between UNITS1+DATA_DIM, UNITS2+UNITS1

const int  LVL_SEQ[2] {1,0};

const int steps = TIMESTEPS;
const int dim = DATA_DIM;
const int sdim = TIMESTEPS*DATA_DIM;
const int un1 = UNITS1;
const int un2 = UNITS2;
const int min_un = UNITS1;
const int max_un = UNITS2;
const int max_mtx = MAX_UNITS*MAX_MCOL;
const int max_vct = 2*MAX_UNITS;
const int max_dmtx = MAX_UNITS*DATA_DIM;
const int max_dvct = DATA_DIM; 
const int min_mcol = UNITS1+DATA_DIM; 
const int max_mcol = UNITS1+UNITS2;


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

dtype row_vector_mul(const dtype w[MAX_MCOL], dtype h_x[MAX_MCOL], dtype b)
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

void input_gate(dtype Wi1[MAX_UNITS/2][MAX_MCOL], dtype bi1[MAX_UNITS/2], dtype Wi2[MAX_UNITS/2][MAX_MCOL], dtype bi2[MAX_UNITS/2], dtype h_xi[MAX_MCOL], dtype out_i1[MAX_UNITS/2], dtype out_i2[MAX_UNITS/2])
{
#pragma HLS ARRAY_PARTITION variable=Wi1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bi1 complete
#pragma HLS ARRAY_PARTITION variable=Wi2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bi2 complete
#pragma HLS ARRAY_PARTITION variable=h_xi complete
	dtype2 i1_t[MAX_UNITS/2];
	dtype2 i2_t[MAX_UNITS/2];
	LOOP_ROW_I:for(int i=0; i<MAX_UNITS/2; ++i) 
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		i1_t[i] = row_vector_mul(Wi1[i], h_xi, bi1[i]);
                out_i1[i] = sigmoid(i1_t[i]);
		i2_t[i] = row_vector_mul(Wi2[i], h_xi, bi2[i]);
                out_i2[i] = sigmoid(i2_t[i]);
	}
	//array_sigmoid(i_t, out_i);
}

void forget_gate(dtype Wf1[MAX_UNITS/2][MAX_MCOL], dtype bf1[MAX_UNITS/2], dtype Wf2[MAX_UNITS/2][MAX_MCOL], dtype bf2[MAX_UNITS/2], dtype h_xf[MAX_MCOL], dtype out_f1[MAX_UNITS/2], dtype out_f2[MAX_UNITS/2])
{
#pragma HLS ARRAY_PARTITION variable=Wf1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bf1 complete
#pragma HLS ARRAY_PARTITION variable=Wf2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bf2 complete
#pragma HLS ARRAY_PARTITION variable=h_xf complete
	dtype2 f1_t[MAX_UNITS/2];
	dtype2 f2_t[MAX_UNITS/2];
	LOOP_ROW_F:for(int i=0; i<MAX_UNITS/2; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		f1_t[i] = row_vector_mul(Wf1[i], h_xf, bf1[i]);
                out_f1[i] = sigmoid(f1_t[i]);
		f2_t[i] = row_vector_mul(Wf2[i], h_xf, bf2[i]);
                out_f2[i] = sigmoid(f2_t[i]);
	}
	//array_sigmoid(f_t, out_f);
}

void newcell_gate(dtype Wc1[MAX_UNITS/2][MAX_MCOL], dtype bc1[MAX_UNITS/2], dtype Wc2[MAX_UNITS/2][MAX_MCOL], dtype bc2[MAX_UNITS/2], dtype h_xc[MAX_MCOL], dtype out_c1[MAX_UNITS/2], dtype out_c2[MAX_UNITS/2])
{
#pragma HLS ARRAY_PARTITION variable=Wc1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bc1 complete
#pragma HLS ARRAY_PARTITION variable=Wc2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bc2 complete
#pragma HLS ARRAY_PARTITION variable=h_xc complete
	dtype2 c1_t[MAX_UNITS/2];
	dtype2 c2_t[MAX_UNITS/2];
	LOOP_ROW_C:for(int i=0; i<MAX_UNITS/2; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		c1_t[i] = row_vector_mul(Wc1[i], h_xc, bc1[i]);
                out_c1[i] = tanh_fx(c1_t[i]);
		c2_t[i] = row_vector_mul(Wc2[i], h_xc, bc2[i]);
                out_c2[i] = tanh_fx(c2_t[i]);
	}
	//array_tanh(c_t, out_c);
}

void output_gate(dtype Wo1[MAX_UNITS/2][MAX_MCOL], dtype bo1[MAX_UNITS/2], dtype Wo2[MAX_UNITS/2][MAX_MCOL], dtype bo2[MAX_UNITS/2], dtype h_xo[MAX_MCOL], dtype out_o1[MAX_UNITS/2], dtype out_o2[MAX_UNITS/2])
{
#pragma HLS ARRAY_PARTITION variable=Wo1 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bo1 complete
#pragma HLS ARRAY_PARTITION variable=Wo2 dim=2 complete
#pragma HLS ARRAY_PARTITION variable=bo2 complete
#pragma HLS ARRAY_PARTITION variable=h_xo complete
	dtype2 o1_t[MAX_UNITS/2];
	dtype2 o2_t[MAX_UNITS/2];
	LOOP_ROW_O:for(int i=0; i<MAX_UNITS/2; ++i)
	{
		#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
		#pragma HLS PIPELINE II = 1
		o1_t[i] = row_vector_mul(Wo1[i], h_xo, bo1[i]);
                out_o1[i] = sigmoid(o1_t[i]);
		o2_t[i] = row_vector_mul(Wo2[i], h_xo, bo2[i]);
                out_o2[i] = sigmoid(o2_t[i]);
	}
	//array_sigmoid(o_t, out_o);
}

dtype dense_vector_mul(const dtype w[MAX_UNITS], dtype h_x[MAX_UNITS], dtype b)
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

extern "C" {
//void krnl_lstm(dtype *data, ap_uint<512> *iWi, dtype *ibi, ap_uint<512> *iWf, dtype *ibf, ap_uint<512> *iWc, dtype *ibc, ap_uint<512> *iWo, dtype *ibo, dtype *iW, dtype *ib, dtype *res, int active_timesteps)
//void krnl_lstm(dtype *data, ap_int<512> *iWi, dtype *ibi, ap_int<512> *iWf, dtype *ibf, ap_int<512> *iWc, dtype *ibc, ap_int<512> *iWo, dtype *ibo, dtype *iW, dtype *ib, dtype *res, int active_timesteps)
void krnl_lstm(ap_int<512> *data, ap_int<512> *iWi, ap_int<512> *ibi, ap_int<512> *iWf, ap_int<512> *ibf, ap_int<512> *iWc, ap_int<512> *ibc, ap_int<512> *iWo, ap_int<512> *ibo, ap_int<512> *iW, ap_int<16> *ib, short int *res, int active_timesteps)
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

	dtype c[MAX_UNITS], h[MAX_UNITS];
	dtype c_f[MAX_UNITS], c_s[MAX_UNITS], c_tmp[MAX_UNITS], c_tanh_tmp[MAX_UNITS], h_tmp[MAX_UNITS];
	dtype x_h[MAX_MCOL];
	dtype f_t[MAX_UNITS/2], i_t[MAX_UNITS/2], c_t[MAX_UNITS/2], o_t[MAX_UNITS/2];
	dtype f_t2[MAX_UNITS/2], i_t2[MAX_UNITS/2], c_t2[MAX_UNITS/2], o_t2[MAX_UNITS/2];

	int actv_timesteps = active_timesteps;

	dtype input[TIMESTEPS][MAX_UNITS];
	#pragma HLS ARRAY_PARTITION variable=input dim=1
	dtype Wi11_reg[UNITS1/2][UNITS1+UNITS2];
	dtype Wi12_reg[UNITS1/2][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wi11_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wi12_reg dim=2 complete
	dtype bi11_reg[UNITS1/2];
	dtype bi12_reg[UNITS1/2];
	#pragma HLS ARRAY_PARTITION variable=bi11_reg complete
	#pragma HLS ARRAY_PARTITION variable=bi12_reg complete
	dtype Wf11_reg[UNITS1/2][UNITS1+UNITS2];
	dtype Wf12_reg[UNITS1/2][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wf11_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wf12_reg dim=2 complete
	dtype bf11_reg[UNITS1/2];
	dtype bf12_reg[UNITS1/2];
	#pragma HLS ARRAY_PARTITION variable=bf11_reg complete
	#pragma HLS ARRAY_PARTITION variable=bf12_reg complete
	dtype Wc11_reg[UNITS1/2][UNITS1+UNITS2];
	dtype Wc12_reg[UNITS1/2][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wc11_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wc12_reg dim=2 complete
	dtype bc11_reg[UNITS1/2];
	dtype bc12_reg[UNITS1/2];
	#pragma HLS ARRAY_PARTITION variable=bc11_reg complete
	#pragma HLS ARRAY_PARTITION variable=bc12_reg complete
	dtype Wo11_reg[UNITS1/2][UNITS1+UNITS2];
	dtype Wo12_reg[UNITS1/2][UNITS1+UNITS2];
	#pragma HLS ARRAY_PARTITION variable=Wo11_reg dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=Wo12_reg dim=2 complete
	dtype bo11_reg[UNITS1/2];
	dtype bo12_reg[UNITS1/2];
	#pragma HLS ARRAY_PARTITION variable=bo11_reg complete
	#pragma HLS ARRAY_PARTITION variable=bo12_reg complete

	dtype W_reg[DATA_DIM][UNITS2];
	#pragma HLS ARRAY_PARTITION variable=W_reg dim=2 complete
	dtype b_reg[DATA_DIM];
	#pragma HLS ARRAY_PARTITION variable=b_reg complete

	dtype2 output[TIMESTEPS][DATA_DIM];
	#pragma HLS ARRAY_PARTITION variable=output dim=1

	//for(int i = 0, x=0, y=0; i< (actv_timesteps*DATA_DIM); i++, y++)
	for(int i = 0, x=0, y=0; i< ((TIMESTEPS*DATA_DIM)/32); i++, y++)
	{
		//#pragma HLS LOOP_TRIPCOUNT min = sdim max = sdim
		#pragma HLS LOOP_TRIPCOUNT min = 2 max = 2
		#pragma HLS PIPELINE II = 1
		ap_int<512> Dt = data[i];
		if (y == DATA_DIM) //Because y dim is smaller else it should have been /32
		{
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
				input[x*(512/16)+j*2+z][y]=(float)d_tmp/FRAQ_CONV;
			}
		}
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
				W_reg[x][y*(512/16)+j*2+z]=(float)W_tmp/FRAQ_CONV;
			}
		}
	}
	for(int i = 0, x = 0 ; i< DATA_DIM; i++, x++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = max_dvct max = max_dvct
		#pragma HLS PIPELINE II = 1
		b_reg[x] = (float)ib[i]/FRAQ_CONV;
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
			//for(int j=0;j<(512/16);j++)
			for(int j=0;j<(512/32);j++)
			{
				//#pragma HLS LOOP_TRIPCOUNT min = 32 max = 32
				#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
				#pragma HLS PIPELINE II = 1
				/*Wi1_reg[x][y*(512/16)+j]=dtype(Wi.range((j+1)*16-1, j*16));
				Wf1_reg[x][y*(512/16)+j]=dtype(Wf.range((j+1)*16-1, j*16));
				Wc1_reg[x][y*(512/16)+j]=dtype(Wc.range((j+1)*16-1, j*16));
				Wo1_reg[x][y*(512/16)+j]=dtype(Wo.range((j+1)*16-1, j*16));*/
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
                                        if(x<UNITS1/2)
					{
						Wi11_reg[x][y*(512/16)+j*2+z]=(float)Wi_tmp/FRAQ_CONV;
						Wf11_reg[x][y*(512/16)+j*2+z]=(float)Wf_tmp/FRAQ_CONV;
						Wc11_reg[x][y*(512/16)+j*2+z]=(float)Wc_tmp/FRAQ_CONV;
						Wo11_reg[x][y*(512/16)+j*2+z]=(float)Wo_tmp/FRAQ_CONV;
					}
					else
					{
						Wi12_reg[x][y*(512/16)+j*2+z]=(float)Wi_tmp/FRAQ_CONV;
						Wf12_reg[x][y*(512/16)+j*2+z]=(float)Wf_tmp/FRAQ_CONV;
						Wc12_reg[x][y*(512/16)+j*2+z]=(float)Wc_tmp/FRAQ_CONV;
						Wo12_reg[x][y*(512/16)+j*2+z]=(float)Wo_tmp/FRAQ_CONV;
					}
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
                                        if(x<2)
					{
						bi11_reg[x*(512/16)+j*2+z]=(float)bi_tmp/FRAQ_CONV;
						bf11_reg[x*(512/16)+j*2+z]=(float)bf_tmp/FRAQ_CONV;
						bc11_reg[x*(512/16)+j*2+z]=(float)bc_tmp/FRAQ_CONV;
						bo11_reg[x*(512/16)+j*2+z]=(float)bo_tmp/FRAQ_CONV;
					}
					else
					{
						bi12_reg[x*(512/16)+j*2+z]=(float)bi_tmp/FRAQ_CONV;
						bf12_reg[x*(512/16)+j*2+z]=(float)bf_tmp/FRAQ_CONV;
						bc12_reg[x*(512/16)+j*2+z]=(float)bc_tmp/FRAQ_CONV;
						bo12_reg[x*(512/16)+j*2+z]=(float)bo_tmp/FRAQ_CONV;
					}
				}
			}
		}
		int pos=0;
		LOOP_TIMESTEPS:for(int k=0; k<actv_timesteps; ++k)
		{
			#pragma HLS LOOP_TRIPCOUNT min = 1 max = steps
			pos = (LVL_SEQ[cl]==0) ? (actv_timesteps-1) : k;
			LOOP_INPUT:for(int i=0; i<MAX_MCOL; ++i)
			{
				#pragma HLS LOOP_TRIPCOUNT min = min_mcol max = max_mcol
				if(cl==0)
					x_h[i] = (i<DATA_DIM) ? input[pos][i] : ((i-DATA_DIM)<MAX_UNITS && k>0) ? h[i-DATA_DIM] : ((dtype) 0.0);
				else if(cl==1)
					x_h[i] = (i<UNITS1) ? input[pos][i] : ((i-UNITS1)<MAX_UNITS && k>0) ? h[i-UNITS1] : ((dtype) 0.0);
			}

			input_gate(Wi11_reg, bi11_reg, Wi12_reg, bi12_reg, x_h, i_t, i_t2);
			forget_gate(Wf11_reg, bf11_reg, Wf12_reg, bf12_reg, x_h, f_t, f_t2);
			newcell_gate(Wc11_reg, bc11_reg, Wc12_reg, bc2_reg, x_h, c_t, c_t2);
			output_gate(Wo11_reg, bo11_reg, Wo12_reg, bo12_reg, x_h, o_t, o_t2);

			LOOP_IN_UNITS:for(int i=0; i<MAX_UNITS; ++i)
			{
				#pragma HLS LOOP_TRIPCOUNT min = min_un max = max_un
				#pragma HLS PIPELINE II = 1
				c_f[i] = (k==0) ? ((dtype) 0.0) : (i<MAX_UNITS/2) ? ((dtype)(f_t[i]*c[i])) : ((dtype)(f_t2[i-(MAX_UNITS/2)]*c[i]));
				c_s[i] = (i<MAX_UNITS/2) ? (c_t[i]*i_t[i]) : (c_t2[i-(MAX_UNITS/2)]*i_t2[i-(MAX_UNITS/2)]);
				c_tmp[i] = c_f[i] + c_s[i];
				c_tanh_tmp[i] = tanh_fx((dtype2)c_tmp[i]);
				h_tmp[i] = (i<MAX_UNITS/2) ? (o_t[i] * c_tanh_tmp[i]) : (o_t2[i-(MAX_UNITS/2)] * c_tanh_tmp[i]);
				c[i] = c_tmp[i];
				h[i] = h_tmp[i];
				input[k][i] = h_tmp[i];
			}
			if(cl==1)
			{
				Dense(W_reg, b_reg, h_tmp, output[k]);
			}
		}
	}
	for(int i = 0, x=0, y=0; i< actv_timesteps*DATA_DIM; i++, y++)
	{
		#pragma HLS LOOP_TRIPCOUNT min = sdim max = sdim
		#pragma HLS PIPELINE II = 1
		if (y == DATA_DIM)
		{
			x++;
			y = 0;
		}
		res[i] = output[x][y]*FRAQ2_CONV;
	}
	/*for(int i = 0, x=0, y=0; i< ((TIMESTEPS*DATA_DIM)/32); i++, y++)
	{
		//#pragma HLS LOOP_TRIPCOUNT min = sdim max = sdim
		#pragma HLS LOOP_TRIPCOUNT min = 2 max = 2
		#pragma HLS PIPELINE II = 1
		ap_int<512> Dto;
		float tmp;
		dtype2 tmp2;
		if (y == DATA_DIM) //Because y dim is smaller else it should have been /32
		{
			x++;
			y = 0;
		}
		for(int j=0;j<(512/32);j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16
			#pragma HLS PIPELINE II = 1
			for(int z=0;z<2;z++)
			{
				tmp = output[x*(512/16)+j*2+z][y].to_float();
				tmp2 = tmp*FRAQ_CONV;
				if(z==0)
					Dto.range(((j+1)*32)-17, j*32)=tmp2;
				else
					Dto.range(((j+1)*32)-1, (j*32)+16)=tmp2;
			}
		}
		res[i] = Dto[i];
	}*/
	/*ap_int<512> Dto;
	for(int i=0; i<((TIMESTEPS*16)/512); i++)
	{
		for(int x=0, y=0; x<(TIMESTEPS/2); x++)
		{
			#pragma HLS LOOP_TRIPCOUNT min = steps max = steps
			#pragma HLS PIPELINE II = 1
			float tmp = output[x+i*(TIMESTEPS/2)][y].to_float();
			float tmp1 = tmp*FRAQ_CONV
			dtype2 tmp2 = tmp1;
			Dto.range(((x+1)*16)-1, x*16)=tmp2;	
		}
		res = Dto;
	}*/
}
}
