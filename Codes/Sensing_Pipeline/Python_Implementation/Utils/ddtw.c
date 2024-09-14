#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>


#define N 3  // Dimension of the arrays

// derivative between two points
double diff_dist(double* a[N], double* b[N]) {
    double da_x = ((a[1][0] - a[0][0]) + (a[2][0] - a[0][0]) / 2) / 2;
    double da_y = ((a[1][1] - a[0][1]) + (a[2][1] - a[0][1]) / 2) / 2;
    double da_z = ((a[1][2] - a[0][2]) + (a[2][2] - a[0][2]) / 2) / 2;
    
    double db_x = ((b[1][0] - b[0][0]) + (b[2][0] - b[0][0]) / 2) / 2;
    double db_y = ((b[1][1] - b[0][1]) + (b[2][1] - b[0][1]) / 2) / 2;
    double db_z = ((b[1][2] - b[0][2]) + (b[2][2] - b[0][2]) / 2) / 2;
    return sqrt((da_x - db_x) * (da_x - db_x) + (da_y - db_y) * (da_y - db_y) + (da_z - db_z) * (da_z - db_z));
}

// Derivative Dynamic Time Warping implementation
void DDTW(double signal_1[][N], int n1, double template_1[][N], int n2, double ddtw[][n2 - 2], int ddtw_traceback[][n2 - 2]) {
    
    // Normalize each axis of the signal: d1 = (d1-average)/std
    double sum = 0.0;
    for(int i = 0; i < n2; i++)
    {
        sum += template_1[i][0];
    }
    double average = sum / n2;
    double std = 0.0;
    for(int i = 0; i < n2; i++)
    {
        std += (template_1[i][0] - average) * (template_1[i][0] - average);
    }
    std = sqrt(std / n2);
    for(int i = 0; i < n2; i++)
    {
        template_1[i][0] = (template_1[i][0] - average) / std;
    }

    // normalize signal[][1]
    sum = 0.0;
    for(int i = 0; i < n2; i++)
    {
        sum += template_1[i][1];
    }
    average = sum / n2;
    std = 0.0;
    for(int i = 0; i < n2; i++)
    {
        std += (template_1[i][1] - average) * (template_1[i][1] - average);
    }
    std = sqrt(std / n2);
    for(int i = 0; i < n2; i++)
    {
        template_1[i][1] = (template_1[i][1] - average) / std;
    }

    // normalize signal[][2]
    sum = 0.0;
    for(int i = 0; i < n2; i++)
    {
        sum += template_1[i][2];
    }
    average = sum / n2;
    std = 0.0;
    for(int i = 0; i < n2; i++)
    {
        std += (template_1[i][2] - average) * (template_1[i][2] - average);
    }
    std = sqrt(std / n2);
    for(int i = 0; i < n2; i++)
    {
        template_1[i][2] = (template_1[i][2] - average) / std;
    }


    assert(n1 != 0 && n2 != 0 && n1 >= 3 && n2 >= 3);
    int n_rows = n1 - 2;
    int n_cols = n2 - 2;

    double *slice_1[N] = {signal_1[0], signal_1[1], signal_1[2]};
    double *slice_2[N] = {template_1[0], template_1[1], template_1[2]};

    ddtw[0][0] = diff_dist(slice_1, slice_2);
    // printf("%.2lf", ddtw[0][0]);
    for (int i = 1; i < n_rows; i++) {
        double *tmp_slice_1[3] = {signal_1[i-1], signal_1[i], signal_1[i+1]};
        double *tmp_slice_2[3] = {template_1[0], template_1[1], template_1[2]};
        ddtw[i][0] = ddtw[i - 1][0] + diff_dist(tmp_slice_1, tmp_slice_2);
        ddtw_traceback[i][0] = 1;
    }

    for (int j = 1; j < n_cols; j++) {
        double *tmp_slice_3[3] = {signal_1[0], signal_1[1], signal_1[2]};
        double *tmp_slice_4[3] = {template_1[j-1], template_1[j], template_1[j+1]};
        ddtw[0][j] = ddtw[0][j - 1] + diff_dist(tmp_slice_3, tmp_slice_4);
        ddtw_traceback[0][j] = 2;
    }

    for (int i = 1; i < n_rows; i++) {
        for (int j = 1; j < n_cols; j++) {
            double temp[3] = {ddtw[i - 1][j - 1], ddtw[i - 1][j], ddtw[i][j - 1]};

            int best_idx = 0;
            for (int k = 1; k < 3; k++) {
                if (temp[k] < temp[best_idx]) {
                    best_idx = k;
                }
            }
            double *tmp_slice_5[3] = {signal_1[i-1], signal_1[i], signal_1[i+1]};
            double *tmp_slice_6[3] = {template_1[j-1], template_1[j], template_1[j+1]};

            ddtw[i][j] = diff_dist(tmp_slice_5, tmp_slice_6) + temp[best_idx];
            ddtw_traceback[i][j] = best_idx;
            // printf("%.2lf\n", ddtw[i][j]);     
        }
    }
}

// Traceback to get the distance
double get_traceback(int n1, int n2, double ddtw[][n2 - 2], int ddtw_traceback[][n2 - 2])
{
    int i = n1-2;
    int j = n2-2;
    i -= 1;
    j -= 1;
    double dis = 0.0;
    int x[1000]; // Assuming a maximum size for x and y, adjust as needed
    int y[1000];
    int count = 0;
    x[count] = i;
    y[count] = j;
    count++;
    while (i != 0 || j != 0) 
    {
        if (i != 0 && j != 0) 
        {
            int idx_i[3] = {i - 1, i - 1, i};
            int idx_j[3] = {j - 1, j, j - 1};
            int idx = (int)ddtw_traceback[i][j];
            i = idx_i[idx];
            j = idx_j[idx];
        } 
        else if (i == 0 && j != 0) {
            j = j - 1;
        } 
        else if (i != 0 && j == 0) {
            i = i - 1;
        } 
        else if (i == 1 && j == 1) {
            i = 0;
            j = 0;
        }
        x[count] = i;
        y[count] = j;
        count++;
    }
    for (int k = 0; k < count; k++) {
        dis += ddtw[x[k]][y[k]];
    }
    dis /= count;
    return dis;
}

//  Load the template from the file
int load_template(int tem_length, int data_points, int axis_num, double data[][data_points][axis_num])
{
    // Replace with the actual path to the file
    FILE *file = fopen("template_36_5_80.txt", "r");
    if(file ==NULL)
    {
       printf("No file!\n");
        // return 0;
    }

    else{
        int layer = 0;
        int row = 0;
        while (layer < tem_length) {
            row = 0;
            while (row < data_points) {
                int col = 0;
                while (col < axis_num && fscanf(file, "%lf", &data[layer][row][col]) != EOF) {
                    // printf("%.4lf\n", data[layer][row][col]);
                    col++;
                }
                row++;
            }
            layer++;
        }
        fclose(file);
    }
    return 0;
}


double DDTW_matching_res(double signal_1[][N], int ang_gran, int dis_gran, int test_p, int gt_points, int axis)
{
    int length = dis_gran * ang_gran;
    double template[length][gt_points][axis];
    load_template(length, gt_points, axis, template);   
    int n1 = test_p; // Replace with the actual size of signal_1
    int n2 = gt_points; // Replace with the actual size of signal_2
    double ddtw[n1 - 2][n2 - 2];
    int ddtw_traceback[n1 - 2][n2 - 2];

    double minRes = INT32_MAX;
    double angle = 0.0;
    int ind = 0;
    // printf("length is %d\n", length);

    // normalize each axis of the signal: d1 = (d1-average)/std
    // only normalize signal[][0]
    double sum = 0.0;
    for(int i = 0; i < n1; i++)
    {
        sum += signal_1[i][0];
    }
    double average = sum / n1;
    double std = 0.0;
    for(int i = 0; i < n1; i++)
    {
        std += (signal_1[i][0] - average) * (signal_1[i][0] - average);
    }
    std = sqrt(std / n1);
    for(int i = 0; i < n1; i++)
    {
        signal_1[i][0] = (signal_1[i][0] - average) / std;
    }

    // normalize signal[][1]
    sum = 0.0;
    for(int i = 0; i < n1; i++)
    {
        sum += signal_1[i][1];
    }
    average = sum / n1;
    std = 0.0;
    for(int i = 0; i < n1; i++)
    {
        std += (signal_1[i][1] - average) * (signal_1[i][1] - average);
    }
    std = sqrt(std / n1);
    for(int i = 0; i < n1; i++)
    {
        signal_1[i][1] = (signal_1[i][1] - average) / std;
    }

    // normalize signal[][2]
    sum = 0.0;
    for(int i = 0; i < n1; i++)
    {
        sum += signal_1[i][2];
    }
    average = sum / n1;
    std = 0.0;
    for(int i = 0; i < n1; i++)
    {
        std += (signal_1[i][2] - average) * (signal_1[i][2] - average);
    }
    std = sqrt(std / n1);
    for(int i = 0; i < n1; i++)
    {
        signal_1[i][2] = (signal_1[i][2] - average) / std;
    }



    for(int i = 0; i < length; i++)
    {
        DDTW(signal_1, n1, template[i], n2, ddtw, ddtw_traceback);
        double dis = get_traceback(n1, n2, ddtw, ddtw_traceback);
        // printf("%.4lf\n", dis);
        if(dis < minRes)
        {
            minRes = dis;
            ind = i;
        }
    }
    // printf("The minimum distance is %.4lf\n", minRes);
    // printf("The index is %d\n", ind);
    angle = (ind / dis_gran) * (360.0 / ang_gran); 
    double dis = (ind % dis_gran) - dis_gran / 2;
    // printf("The distance is %.4lf\n", dis);
    // printf("The angle is %.4lf\n", angle);
    return angle;
}

// Test the DDTW algorithm
int main()
{ 

    const int tem_length = 180;
    const int gt_points = 20;
    const int axis_num = 3;
    clock_t start, end;
    // Test data
    double test[][N] = 
    {{-21.45, 22.95, 29.04}, {-21.15, 23.25, 27.346}, {-22.2, 22.95, 28.314}, {-21.0, 23.7, 28.314}, {-24.75, 20.4, 29.766}, {-26.25, 18.45, 31.702}, {-30.45, 17.7, 34.848}, 
    {-41.1, 15.15, 43.802}, {-57.6, 12.75, 58.806}, {-88.65, 27.3, 80.102}, {-135.6, 78.15, 87.12}, {-172.5, 155.55, 2.662}, {-154.2, 138.9, -146.41}, {-97.95, 20.7, -149.072}, 
    {-58.95, -15.75, -68.97}, {-37.35, -5.4, -13.068}, {-28.2, 7.65, 4.598}, {-23.1, 13.2, 13.068}, {-20.1, 21.15, 17.424}, {-17.55, 22.5, 20.57}};
    start = clock();

    int ang_gran = 36;
    int dis_gran = 21;
    int length = ang_gran * dis_gran;
    int test_p = 20;
    DDTW_matching_res(test, ang_gran, dis_gran, test_p, gt_points, axis_num);
    end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Test duration: ");
    printf("%lf s\n", duration);
    return 0;
}