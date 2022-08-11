//---------------------------------//
//- Author: Stamoulias Ioannis    -//
//- email: jstamoulias@gmail.com  -//
//---------------------------------//

void device_init(char* krnl_name);
void lstm_io(int time_steps, int max_steps, int features, int iunits, int ounits, int mx_batch);
void lstm_accel(short int *input, short int *output, short int *Wi, short int *bi, short int *Wf, short int *bf, short int *Wc, short int *bc, short int *Wo, short int *bo, short int *W, short int *b, int time_steps, int max_steps, int init, int features, int iunits, int ounits, int mx_batch);
void lstm_end();
short int fixp(float value, int fracbits);
