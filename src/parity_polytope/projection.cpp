////////////////////////////////////////////////////////////////////////
//
// Disclaimer:
// This code and instructions are modified versions of "ADMM Decoder"
// downloaded from https://sites.google.com/site/xishuoliu/codes.
// Full credits goes to original authors.
//
////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;
struct NODE 
{
	int index;
	double value;
}node;
// comparer for nodes. needed for qsort

bool sort_compare_de (const NODE& a, const NODE& b)
{
	return (a.value > b.value);
}

// Total projection, need to search on alpha
void FastProjection(NODE v[], double* results, int length)
{
	double zero_tol_offset = 1e-10;
	bool AllZero = true, AllOne = true;
	for(int i = 0; i < length; i++)
	{
		if(v[i].value > 0)
			AllZero = false;
		if(v[i].value <= 1)
			AllOne = false;
	}
	if(AllZero) // exit if the vector has all negative values, which means that the projection is all zero vector
	{
		for(int i = 0;i < length; i++)
			results[i] = 0;
		return;
	}
	if (AllOne && (length%2 == 0)) // exit if the vector should be all one vector.
	{
		for(int i = 0;i < length; i++)
			results[i] = 1;
		return;
	}

	NODE *zSort = new NODE[length]; // vector that save the sorted input vector
	for(int i = 0;i < length; i++) // keep the original indices while sorting, 
	{
		zSort[i].index = v[i].index;
		zSort[i].value = v[i].value;
	}
	sort(zSort, zSort + length, sort_compare_de);//sort input vector, decreasing order

	
	int clip_idx = 0, zero_idx= 0;
	double constituent = 0;
	NODE *zClip = new NODE[length]; 
	for(int i = 0;i < length; i++)// project on the [0,1]^d cube
	{
		zClip[i].value = min(max(zSort[i].value, 0.0),1.0);
		zClip[i].index = zSort[i].index;
		constituent += zClip[i].value;
	} 
	int r = (int)floor(constituent);
	if (r & 1)
		r--;
	// calculate constituent parity

	// calculate sum_Clip = $f_r^T z$
	double sum_Clip = 0;
	for(int i = 0; i < r+1; i++)
		sum_Clip += zClip[i].value;
	for(int i = r + 1; i < length; i++)
		sum_Clip -= zClip[i].value;
	

	if (sum_Clip <= r) // then done, return projection, beta = 0
	{
		for(int i = 0; i < length; i++)
			results[zClip[i].index] = zClip[i].value;
		delete [] zSort; delete [] zClip;
		return;
	}

	double beta = 0;
	double beta_max = 0;
	if (r + 2 <= length)
		beta_max = (zSort[r].value - zSort[r+1].value)/2; // assign beta_max
	else
		beta_max = zSort[r].value;

	NODE *zBetaRep = new NODE[length]; 

	// merge beta, save sort
	int left_idx = r, right_idx = r + 1;
	int count_idx = 0;
	while(count_idx < length)
	{
		double temp_a, temp_b;
		if(left_idx < 0)
		{
			while(count_idx < length)
			{
				zBetaRep[count_idx].index = right_idx;
				zBetaRep[count_idx].value = -zSort[right_idx].value;
				right_idx++;count_idx++;
			}
			break;
		}
		if(right_idx >= length)
		{
			while(count_idx < length)
			{
				zBetaRep[count_idx].index = left_idx;
				zBetaRep[count_idx].value = zSort[left_idx].value - 1;
				left_idx--;count_idx++;
			}
			break;
		}
		temp_a = zSort[left_idx].value - 1; temp_b = -zSort[right_idx].value;
		if(temp_a > temp_b || left_idx < 0)
		{
			zBetaRep[count_idx].index = right_idx;
			zBetaRep[count_idx].value = temp_b;
			right_idx++;count_idx++;
		}
		else
		{
			zBetaRep[count_idx].index = left_idx;
			zBetaRep[count_idx].value = temp_a;
			left_idx--;count_idx++;
		}
	}
	
    //initialize "waterfilling"
	for(int i = 0; i < length; i++)
	{
		if (zSort[i].value > 1)
			clip_idx++;
		if (zSort[i].value >= 0 - zero_tol_offset)
			zero_idx++;
	}
	clip_idx--;
	bool first_positive = true, below_beta_max = true;
	int idx_start = 0, idx_end = 0;
	for(int i = 0;i < length; i++)
	{
		if(zBetaRep[i].value < 0 + zero_tol_offset && first_positive)
		{
			idx_start++;
		}
		if(zBetaRep[i].value < beta_max && below_beta_max)
		{
			idx_end++;
		}
	}
	idx_end--;
	double active_sum = 0;
	for(int i = 0;i < length; i++)
	{
		if(i > clip_idx && i <= r)
			active_sum += zSort[i].value;
		if(i > r && i < zero_idx)
			active_sum -= zSort[i].value;
	}
	double total_sum = 0;
	total_sum = active_sum + clip_idx + 1;

    
	int previous_clip_idx, previous_zero_idx;
	double previous_active_sum;
	bool change_pre = true;
	previous_clip_idx = clip_idx;
	previous_zero_idx = zero_idx;
	previous_active_sum = active_sum;
	for(int i = idx_start; i <= idx_end; i++)
	{
		if(change_pre)
		{
			// save previous things
			previous_clip_idx = clip_idx;
			previous_zero_idx = zero_idx;
			previous_active_sum = active_sum;
		}
		change_pre = false;

		beta = zBetaRep[i].value;
		if(zBetaRep[i].index <= r)
		{
			clip_idx--;
			active_sum += zSort[zBetaRep[i].index].value;
		}
		else
		{
			zero_idx++;
			active_sum -= zSort[zBetaRep[i].index].value;
		}
		if (i < length - 1)
		{
			if (beta != zBetaRep[i + 1].value)
			{
				total_sum = (clip_idx+1) + active_sum - beta * (zero_idx - clip_idx - 1);
				change_pre = true;
				if(total_sum < r)
					break;
			}
			else
			{// actually you shouldn't go here. just continue
				if(0)
					cout<<"here here"<<endl;
			}
		}
		else if (i == length - 1)
		{
			total_sum = (clip_idx + 1)  + active_sum - beta * (zero_idx - clip_idx - 1);
			change_pre = true;
		}
	}
	if (total_sum > r)
	{
		beta = -(r - clip_idx - 1 - active_sum)/(zero_idx - clip_idx - 1); // set beta for the current beta break point
	}
	else
	{
		beta = -(r - previous_clip_idx - 1 - previous_active_sum)/(previous_zero_idx - previous_clip_idx - 1);// set beta for the previous break point
	}
	for(int i = 0;i < length; i++) // assign values for projection results using beta
	{
		if (i <= r)
		{
			results[zSort[i].index] = min(max(zSort[i].value - beta, 0.0),1.0);
		}
		else
		{
			results[zSort[i].index] = min(max(zSort[i].value + beta, 0.0),1.0);
		}

	}
	delete [] zSort; delete [] zClip; delete [] zBetaRep;
}


// single vector input
extern "C" void proj_vec(int length, double *arr_in, double *arr_out)
{
	NODE *v = new NODE[length];
	for (int i =0; i < length; i++)
	{
		v[i].index = i;
		v[i].value = arr_in[i];
	}
	FastProjection(v, arr_out, length);
	delete [] v;
}


// Input is Python sparse CSR like matrix. Do projection for each row.
extern "C" void proj_csr(int length_indptr, int *indptr, double *data_in, double *data_out)
{
	for (int i = 0; i < length_indptr-1; i++)
	{
		int start_ind = indptr[i];
		int num_elements = indptr[i+1]-indptr[i];
		// printf ("%d[%d], ", start_ind, num_elements);
    	proj_vec(num_elements, data_in+start_ind, data_out+start_ind);
	}
}
