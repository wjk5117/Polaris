/*
 * SPDX-FileCopyrightText: 2021 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Unlicense OR CC0-1.0
 */



/****************************************************************************
*
* This demo showcases BLE GATT client. It can scan BLE devices and connect to one device.
* Run the gatt_server demo, the client demo will automatically connect to the gatt_server demo.
* Client demo will enable gatt_server's notify after connection. The two devices will then exchange
* data.
*
****************************************************************************/

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include "nvs.h"
#include "nvs_flash.h"

#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gattc_api.h"
#include "esp_gatt_defs.h"
#include "esp_bt_main.h"
#include "esp_gatt_common_api.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

#include "time.h"
#include "sdkconfig.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_spiffs.h"
#include "esp_log.h"


#define TAG "spiffs"

#define GATTC_TAG "GATTC_DEMO"

#define PROFILE_NUM      1
#define PROFILE_A_APP_ID 0
#define INVALID_HANDLE   0



#define BYTES_PER_LINE 36

#define WND 3
#define WND_D 3



// DDTW algorithm
#define N 3  // Dimension of the arrays
double template[8][20][3];

double diff_dist(double* a[N], double* b[N]) {
    double da_x = ((a[1][0] - a[0][0]) + (a[2][0] - a[0][0]) / 2) / 2;
    double da_y = ((a[1][1] - a[0][1]) + (a[2][1] - a[0][1]) / 2) / 2;
    double da_z = ((a[1][2] - a[0][2]) + (a[2][2] - a[0][2]) / 2) / 2;
    
    double db_x = ((b[1][0] - b[0][0]) + (b[2][0] - b[0][0]) / 2) / 2;
    double db_y = ((b[1][1] - b[0][1]) + (b[2][1] - b[0][1]) / 2) / 2;
    double db_z = ((b[1][2] - b[0][2]) + (b[2][2] - b[0][2]) / 2) / 2;
    return sqrt((da_x - db_x) * (da_x - db_x) + (da_y - db_y) * (da_y - db_y) + (da_z - db_z) * (da_z - db_z));

    // return sqrt(pow((da_z - db_z), 2) + pow(sqrt((da_x + da_y) *(da_x + da_y)) - sqrt((db_x + db_y) * (db_x + db_y)), 2));

}

void DDTW(double signal_1[][N], int n1, double template_1[][N], int n2, double ddtw[][n2 - 2], int ddtw_traceback[][n2 - 2]) {
    
    // only normalize signal[][0]
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


int load_template(int tem_length, int data_points, int axis_num, double data[][data_points][axis_num])
{

    esp_vfs_spiffs_conf_t config = {
      .base_path = "/spiffs",
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = true,
    };
    esp_vfs_spiffs_register(&config);

    FILE *file = fopen("/spiffs/template_8_1.txt", "r");
    if(file ==NULL)
    {
        ESP_LOGE(TAG,"File does not exist!");
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
        esp_vfs_spiffs_unregister(NULL);
    }
    return 0;
}

const int n1 = 20; // Replace with the actual size of signal_1
const int n2 = 20; // Replace with the actual size of signal_2
double ddtw[20 - 2][20 - 2];
int ddtw_traceback[20 - 2][20 - 2];
double DDTW_matching_res(double signal_1[][N], int ang_gran, int dis_gran, int test_p, int gt_points, int axis)
{
    int length = dis_gran * ang_gran;

    load_template(length, gt_points, axis, template);   

    // double ddtw[n1 - 2][n2 - 2];
    // int ddtw_traceback[n1 - 2][n2 - 2];

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
    angle = (ind / dis_gran) * 
    (360.0 / ang_gran); 
    double dis = (ind % dis_gran) - dis_gran / 2;
    // printf("The distance is %.4lf\n", dis);
    // printf("The angle is %.4lf\n", angle);
    return angle;
}




// localize algorithm

const double pi = 3.14159265358979323846;
// Parameter we want to sovle
const int N_2 = 1;

/*
Functions to be used in the main function
*/
// Check if the input is a valid field input
int check_field_input(const char *inp, const char *origin);
// Transform cartesian coordinates to cylindrical coordinates
void cart_to_cyl_coordinates(double observer[3], double result[3]);
// Transform cylindrical coordinates to cartesian coordinates
void cyl_field_to_cart(double phi, double Br, double Bphi, double results[2]);
// Function to calculate the simplified magnetic field
void B_simplified(double x, double y, double h, double res[3]);
// Function to calculate the objective function
double objective_function(double r[], double amp1[3], double amp2[3], double theta, double l, double h);
int constraint(double x[], double l);
void nelder_mead_minimization(double x[], double amp1[3], double amp2[3], double theta, double l, double h);
// ComputeD
double ComputeD(double amp1[3], double amp2[3], double theta, int index1, int index2, double h, double l, double sol[N_2]);
// first kind elliptic integral
double ellipe(double k, int N_2);
// second kind elliptic integral
double ellipk(double k, int N_2);
double cel0(double kc, double p, double c, double s);
// fieldH_cylinder_diametral
void fieldH_cylinder_diametral(double z0, double r, double phi, double z, double res[3]);
// magnetic_field_cylinder
void magnetic_field(double observer[3], double final_res[3]);

double magnetization[3] = {0, 100000, 0};
double dimension[2] = {4, 1};
// Check if the input is a valid field input
// int check_field_input(const char *inp, const char *origin) 
// {
//     if (strcmp(inp, "B") == 0) {
//         return 1;  // True
//     }
//     if (strcmp(inp, "H") == 0) {
//         return 0;  // False
//     }

//     fprintf(stderr, "%s input can only be `field='B'` or `field='H'`.\n"
//                     "Instead received %s.\n", origin, inp);
//     exit(EXIT_FAILURE);
// }

// Transform cartesian coordinates to cylindrical coordinates
void cart_to_cyl_coordinates(double observer[3], double result[3]) 
{
    double x = observer[0];
    double y = observer[1];
    double z = observer[2];
    double r = sqrt(x * x + y * y);
    double phi = atan2(y, x);
    result[0] = r;
    result[1] = phi;
    result[2] = z;
}

// Transform cylindrical coordinates to cartesian coordinates
void cyl_field_to_cart(double phi, double Br, double Bphi, double results[2])
{
    // transform Br, Bphi to Bx, By
    results[0] = Br * cos(phi) - Bphi * sin(phi);
    results[1] = Br * sin(phi) + Bphi * cos(phi);
}

// Function to calculate the simplified magnetic field
void B_simplified(double x, double y, double h, double res[3]) 
{
    double magnetization[3] = {0, 2000000, 0};
    double dimension[2] = {2, 1};
    double observer[3] = {x, y, h};
    magnetic_field(observer, res);
}


// Function to calculate the objective function
double objective_function(double r[], double amp1[3], double amp2[3], double theta, double l, double h) 
{
    // r2 = l - r1;
    double r1 = r[0];
    double d = 10;
    double r2 = 2*l-r1 - d;
    double B1_res[3], B2_res[3];
    B_simplified(-r1 * cos(theta), r1 * sin(theta), h, B1_res);
    double B1x = B1_res[0], B1y = B1_res[1], B1z = B1_res[2];

    // double B1x = amp1[0], B1y = amp1[1], B1z = amp1[2];

    B_simplified(r2 * cos(theta), -r2 * sin(theta), h, B2_res);
    // double B2x = B2_res[0], B2y = B2_res[1], B2z = B2_res[2];
    double B2x = amp2[0], B2y = amp2[1], B2z = amp2[2];

    // double eq1 = r1 + r2 - l;
    double eq2 = sqrt(B1x * B1x + B1y * B1y + B1z * B1z) / sqrt(B2x * B2x + B2y * B2y + B2z * B2z) -
                 sqrt(amp1[0] * amp1[0] + amp1[1] * amp1[1] + amp1[2] * amp1[2]) /
                 sqrt(amp2[0] * amp2[0] + amp2[1] * amp2[1] + amp2[2]* amp2[2]);
    return eq2 * eq2;
}

// Constraint function
int constraint(double x[], double l) {
    // Check if x satisfies the constraints
    if (x[0] >= 0 && x[0] <= l) {
        return 1;  // Constraints satisfied
    } else {
        return 0;  // Constraints violated
    }
}

// Nelder-Mead optimization algorithm
void nelder_mead_minimization(double x[], double amp1[3], double amp2[3], double theta, double l, double h) {
    double simplex[N_2+1][N_2], f[N_2+1];
    double alpha = 1.0, gamma = 1.0, rho = 0.5, sigma = 0.5;
    double epsilon = 1e-2;  // Tolerance for convergence
    int iter, max_iter =3;
    
    // Initialize simplex with initial guess x
    for (int i = 0; i <= N_2; i++) {
        for (int j = 0; j < N_2; j++) {
            simplex[i][j] = x[j];
        }
        if (i > 0) {
            simplex[i][i-1] += 1.0;
        }
    }
    
    iter = 0;
    do {
        // Evaluate objective function for each vertex of the simplex
        for (int i = 0; i <= N_2; i++) {
            f[i] = objective_function(simplex[i], amp1, amp2, theta, l, h);
        }
        
        // Sort simplex vertices based on objective function values
        for (int i = 0; i <= N_2; i++) {
            for (int j = i+1; j <= N_2; j++) {
                if (f[j] < f[i]) {
                    double temp = f[i];
                    f[i] = f[j];
                    f[j] = temp;
                    for (int k = 0; k < N_2; k++) {
                        temp = simplex[i][k];
                        simplex[i][k] = simplex[j][k];
                        simplex[j][k] = temp;
                    }
                }
            }
        }
        
        // Check convergence
        double diff = fabs(f[N_2] - f[0]);
        if (diff < epsilon) {
            break;  // Convergence achieved
        }
        
        // Compute centroid of best N vertices
        double centroid[N_2];
        for (int i = 0; i < N_2; i++) {
            centroid[i] = 0.0;
            for (int j = 0; j < N_2; j++) {
                centroid[i] += simplex[j][i];
            }
            centroid[i] /= N_2;
        }
        
        // Reflection
        double xr[N_2];
        for (int i = 0; i < N_2; i++) {
            xr[i] = centroid[i] + alpha * (centroid[i] - simplex[N_2][i]);
        }
        if (constraint(xr, l)) {
            double fr = objective_function(xr, amp1, amp2, theta, l, h);
            if (fr < f[0]) {
                for (int i = 0; i < N_2; i++) {
                    simplex[N_2][i] = xr[i];
                }
                f[N_2] = fr;
                continue;
            }
        }
        
        // Expansion
        if (objective_function(xr, amp1, amp2, theta, l, h) < f[N_2-1]) {
            double xe[N_2];
            for (int i = 0; i < N_2; i++) {
                xe[i] = centroid[i] + gamma * (xr[i] - centroid[i]);
            }
            if (constraint(xe, l)) {
                double fe = objective_function(xe, amp1, amp2, theta, l, h);
                if (fe < f[N_2]) {
                    for (int i = 0; i < N_2; i++) {
                        simplex[N_2][i] = xe[i];
                    }
                    f[N_2] = fe;
                    continue;
                }
            }
        }
        
        // Contraction
        double xc[N_2];
        for (int i = 0; i < N_2; i++) {
            xc[i] = centroid[i] + rho * (simplex[N_2][i] - centroid[i]);
        }
        if (constraint(xc, l)) {
            double fc = objective_function(xc, amp1, amp2, theta, l, h);
            if (fc < f[N_2]) {
                for (int i = 0; i < N_2; i++) {
                    simplex[N_2][i] = xc[i];
                }
                f[N_2] = fc;
                continue;
            }
        }
        
        // Shrink
        for (int i = 1; i <= N_2; i++) {
            for (int j = 0; j < N_2; j++) {
                simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
            }
            if (!constraint(simplex[i], l)) {
                f[i] = INFINITY;  // Set objective function value to infinity for infeasible solutions
            } else {
                f[i] = objective_function(simplex[i], amp1, amp2, theta, l, h);
            }
        }
        
        iter++;
    } while (iter < max_iter);
    
    // Update x with the best vertex of the simplex
    for (int i = 0; i < N_2; i++) {
        x[i] = simplex[0][i];
    }
}


double ComputeD(double amp1[3], double amp2[3], double theta, int index1, int index2, double h, double l, double sol[N_2]) 
{
    index1 += 1;
    index2 += 1;
    // printf("Detect one magnet!\n");
    // printf("Sensor: %d, amplitude_x: %.2f, amplitude_y: %.2f, amplitude_z: %.2f, amplitude_tol: %.2f\n",
    //        index1, amp1[0], amp1[1], amp1[2], sqrt(amp1[0] * amp1[0] + amp1[1] * amp1[1] + amp1[2] * amp1[2]));
    // printf("Sensor: %d, amplitude_x: %.2f, amplitude_y: %.2f, amplitude_z: %.2f, amplitude_tol: %.2f\n",
    //        index2, amp2[0], amp2[1], amp2[2], sqrt(amp2[0] * amp2[0] + amp2[1] * amp2[1] + amp2[2] * amp2[2]));
    
    // TBD: Solve for d1 and d2
    sol[0] = l/2;
    nelder_mead_minimization(sol, amp1, amp2, theta, l, h);
    double d = ((int)(sol[0] * 100 + 0.5)) / 100.0;  // Round to two decimal places
    double res = 0;
    if (index1 < index2)
    {
        // printf("Distance to Sensor %d: %.2f mm\n", index1, d);
        res = (index1 - 1) * l + d;
        // printf("Distance to Sensor %d: %.2f mm\n", 1, res);
    }
    else
    {
        // printf("Distance to Sensor %d: %.2f mm\n", index2, l - d);
        res = index2 * l - d;
        // printf("Distance to Sensor %d: %.2f mm\n", 1, res);
    }
    // printf("Solve in %f seconds\n", (double)(clock() - cur_time) / CLOCKS_PER_SEC);
    // printf("\n"); 
    return res;
}


// 第一类完全椭圆积分
double ellipe(double k, int Ne) {
    double a = 0.0;   // 积分下限
    double b = pi/2; // 积分上限
    double h = (b - a) / Ne;  // 步长

    double integral = 0.0;
    for (int i = 0; i <= Ne; ++i) 
    {
        double x = a + i * h;
        double term = h * sqrt(1.0 - k * sin(x) * sin(x));
        if (i == 0 || i == N_2)
            term *= 0.5;
        integral += term;
    }
    return integral;
}

// 第二类完全椭圆积分
double ellipk(double k, int Ne) 
{
    double a = 0.0;   // 积分下限
    double b = pi/2; // 积分上限
    double h = (b - a) / Ne;  // 步长

    double integral = 0.0;
    for (int i = 0; i <= Ne; ++i) {
        double x = a + i * h;
        double term = h / sqrt(1.0 - k * sin(x) * sin(x));
        if (i == 0 || i == N_2)
            term *= 0.5;
        integral += term;
    }
    return integral;
}


double cel0(double kc, double p, double c, double s)
{
    double errtol = 0.000001;
    double k = fabs(kc);
    double pp = p;
    double cc = c;
    double ss = s;
    double em = 1.0;
    if (p > 0)
    {
        pp = sqrt(p);
        ss = s / pp;
    }
    else
    {
        double f = kc * kc;
        double q = 1.0 - f;
        double g = 1.0 - pp;
        f = f - pp;
        q = q * (ss - c * pp);
        pp = sqrt(f / g);
        cc = (c - ss) / g;
        ss = -q / (g * g * pp) + cc * pp;
    }
    double f = cc;
    cc = cc + ss / pp;
    double g = k / pp;
    ss = 2 * (ss + f * g);
    pp = g + pp;
    g = em;
    em = k + em;
    double kk = k;
    while (fabs(g - k) > g * errtol)
    {
        k = 2 * sqrt(kk);
        kk = k * em;
        f = cc;
        cc = cc + ss / pp;
        g = kk / pp;
        ss = 2 * (ss + f * g);
        pp = g + pp;
        g = em;
        em = k + em;
    }
    return (pi / 2) * (ss + cc * em) / (em * (em + pp));
}

void fieldH_cylinder_diametral(double z0, double r, double phi, double z, double res[3])
{
    double Hr = 0, Hphi = 0, Hz = 0;
    
    // compute repeated quantities for all cases
    double zp = z + z0;
    double zm = z - z0;
    double zp2 = zp * zp;
    double zm2 = zm * zm;
    double r2 = r*r;

    // case small_r: numerical instability of general solution
    bool mask_small_r = r < 0.05;
    bool mask_general = !mask_small_r;
    if (mask_general){
        double rp = r + 1;
        double rm = r - 1;
        double rp2 = rp * rp;
        double rm2 = rm * rm;
        double ap2 = zp2 + rm*rm;
        double am2 = zm2 + rm*rm;
        double ap = sqrt(ap2);
        double am = sqrt(am2);
        double argp = -4 * r / ap2;
        double argm = -4 * r / am2;
        double argc = -4 * r / rm2;
        double one_over_rm = 1 / rm;

        const int Ne = 200;
        double elle_p = ellipe(argp, Ne);
        double elle_m = ellipe(argm, Ne);

        double ellk_p = ellipk(argp, Ne);
        double ellk_m = ellipk(argm, Ne);
        // elliptic_Pi
        double ellpi_p = cel0(sqrt(1 - argp), 1 - argc, 1, 1);
        double ellpi_m = cel0(sqrt(1 - argm), 1 - argc, 1, 1);
        // compute fields
        Hr = (
            -cos(phi)
            / (4 * pi * r2)
            * (
                -zm * am * elle_m
                + zp * ap * elle_p
                + zm / am * (2 + zm2) * ellk_m
                - zp / ap * (2 + zp2) * ellk_p
                + (zm / am * ellpi_m - zp / ap * ellpi_p) * rp * (r2 + 1) * one_over_rm
            )
        );
        Hphi = (
            sin(phi)
            / (4 * pi * r2)
            * (
                +zm * am * elle_m
                - zp * ap * elle_p
                - zm / am * (2 + zm2 + 2 * r2) * ellk_m
                + zp / ap * (2 + zp2 + 2 * r2) * ellk_p
                + zm / am * rp2 * ellpi_m
                - zp / ap * rp2 * ellpi_p
            )
        );

        Hz = (
            -cos(phi)
            / (2 * pi * r)
            * (
                + am * elle_m
                - ap * elle_p
                - (1 + zm2 + r2) / am * ellk_m
                + (1 + zp2 + r2) / ap * ellk_p
            )
        );
    }

    res[0] = Hr;
    res[1] = Hphi;
    res[2] = Hz; 
}


void magnetic_field(double observer[3], double final_res[3])
{
    // int bh = check_field_input(field, "magnet_cylinder_field()");
    // transform to Cy CS --------------------------------------------
    int bh = 1;
    double cyl_observer[3];
    cart_to_cyl_coordinates(observer, cyl_observer);
    double r = cyl_observer[0];
    double phi = cyl_observer[1];
    double z = cyl_observer[2];
    double r0 = dimension[0]/2, z0 = dimension[1]/2;
    r = r / r0;
    z = z / r0;
    z0 = z0 / r0;

    // allocate field vectors ----------------------------------------
    double Br = 0, Bphi = 0, Bz = 0;

    double tolerance=1e-15;
     // on Cylinder hull plane
    bool m0 = fabs(r - 1.0) < tolerance;
    // on top or bottom plane
    bool m1 = fabs(fabs(z) - z0) < tolerance;
    // in-between top and bottom plane
    bool m2 = fabs(z) <= z0;
    // inside Cylinder hull plane
    bool m3 = r <= 1.0;

    // special case: mag = 0
    bool mask_0 = fabs(magnetization[0]-0.0) < tolerance && fabs(magnetization[1]-0.0) < tolerance && fabs(magnetization[2]-0.0) < tolerance;
    // special case: on Cylinder edge
    bool mask_edge = m0 & m1;
    // general case
    bool mask_gen = !mask_0 && !mask_edge;

    double magx = magnetization[0], magy = magnetization[1], magz = magnetization[2];
    bool mask_tv = fabs(magx-0.0) > tolerance || fabs(magy-0.0) > tolerance;
    bool mask_ax = fabs(magz-0.0) > tolerance;
    bool mask_inside = m2 && m3;
    // general case masks
    mask_tv = mask_tv && mask_gen;
    mask_ax = mask_ax && mask_gen;
    mask_inside = mask_inside & mask_gen;

    // transversal magnetization contributions -----------------------
    if (mask_tv) {
        double magxy = sqrt(magx*magx + magy*magy);
        double tetta = atan2(magy, magx);
        // printf("magxy: %f, tetta: %f\n", magxy, tetta);
        double b_tv[3];
        fieldH_cylinder_diametral(z0, r, phi- tetta, z, b_tv);

        double br_tv = b_tv[0], bphi_tv = b_tv[1], bz_tv = b_tv[2];
        Br += magxy * br_tv;
        Bphi += magxy * bphi_tv;
        Bz += magxy * bz_tv;
    }

    // transform field to cartesian CS -------------------------------
    double B_results[2];
    cyl_field_to_cart(phi, Br, Bphi, B_results);
    double Bx = B_results[0];
    double By = B_results[1];

    // add/subtract Mag when inside for B/H --------------------------
    if (bh)
    {
        final_res[0] = Bx;
        final_res[1] = By;
        final_res[2] = Bz;
    }
    else
    {
        final_res[0] = Bx * 10 / 4 / pi;
        final_res[1] = By * 10 / 4 / pi;
        final_res[2] = Bz * 10 / 4 / pi;
    }
}







// 定义结构体
struct DynamicArray
{
    int length;
    double *data;
};

struct Point3D 
{
    double x;
    double y;
    double z;
    double total;
};

struct SensorArray
{
    struct Point3D *data;
    size_t size;
    size_t capacity; 
};

struct MagnetInfo
{
    int cur_n;
    int index_flag[3];
    struct Point3D tmp_data[3];
};

void initDynamicArray(struct SensorArray *arr, size_t initialCapacity) 
{
    arr->data = malloc(initialCapacity * sizeof(struct Point3D));
    if (arr->data == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    arr->size = 0;
    arr->capacity = initialCapacity;
}


void appendToDynamicArray(struct SensorArray *arr, double x, double y, double z, double total) 
{
    if (arr->size == arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, arr->capacity * sizeof(struct Point3D));
        if (arr->data == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }
    }
    arr->data[arr->size].x = x;
    arr->data[arr->size].y = y;
    arr->data[arr->size].z = z;
    arr->data[arr->size].total = total;
    arr->size++;
}

void freeDynamicArray(struct SensorArray *arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

double gaussian(double x, double pos, double wid) 
{
    return exp(-pow((x - pos) / (0.60056120439323 * wid), 2));
}

void normalize(double *array, int size) 
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += array[i];
    for (int i = 0; i < size; i++)
        array[i] /= sum;
}

double SmoothVector_r[WND];
double SmoothVector_d[WND_D];
int num = 3;
double slope_thr[3];
double LastRAW[3];
int S_flag[3];
double delta_thrd[3];
double amp_thrd[3];
bool AuxiliaryFlag[3];
bool flag_mag_det = false;

double start_raw_amp_x[3];
double start_raw_amp_y[3];
double start_raw_amp_z[3];

struct DynamicArray slope_list[3];
struct DynamicArray raw_list[3];
struct SensorArray sensors[3];
struct SensorArray smooth_sensors[3];
struct SensorArray derivative_sensors[3];
struct SensorArray smooth_derivative_sensors[3];
struct Point3D amp_tmp_list[3];

int cnt = 0;
int cnt_tol = 0;

int first = 0;
int no = 0;
int flag = 1;
int cur_cnt = 0;

double sensor_tol_data[3];
double sensor_x_data[3];
double sensor_y_data[3];
double sensor_z_data[3];

double amp_tol = 0.0;
double amp_tol_ddtw[20][3];
struct MagnetInfo magnetInfo;

static const char remote_device_name[] = "Bluefruit52"; // "ESP_GATTS_DEM
static bool connect    = false;
static bool get_server = false;
static esp_gattc_char_elem_t *char_elem_result   = NULL;
static esp_gattc_descr_elem_t *descr_elem_result = NULL;

/* Declare static functions */
static void esp_gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param);
static void esp_gattc_cb(esp_gattc_cb_event_t event, esp_gatt_if_t gattc_if, esp_ble_gattc_cb_param_t *param);
static void gattc_profile_event_handler(esp_gattc_cb_event_t event, esp_gatt_if_t gattc_if, esp_ble_gattc_cb_param_t *param);

static uint8_t gatts_service_uuid128[ESP_UUID_LEN_128] = {0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3,0xb5, 0x01, 0x00, 0x40, 0x6e};

//Server UUID
static esp_bt_uuid_t remote_filter_service_uuid = {
    .len = ESP_UUID_LEN_128,
    /* LSB <--------------------------------------------------------------------------------> MSB */
    .uuid = {.uuid128 = {0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3,0xb5, 0x01, 0x00, 0x40, 0x6e},},
};
//notify UUID
static esp_bt_uuid_t remote_filter_char_uuid = {
    .len = ESP_UUID_LEN_128,
	/* LSB <--------------------------------------------------------------------------------> MSB */
    .uuid = {.uuid128 = {0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3,0xb5, 0x03, 0x00, 0x40, 0x6e},},
};
//Characteristic UUID
static esp_bt_uuid_t notify_descr_uuid = {
    .len = ESP_UUID_LEN_16,
    .uuid = {.uuid16 = ESP_GATT_UUID_CHAR_CLIENT_CONFIG,},
};


// static esp_bt_uuid_t remote_filter_service_uuid = {
//     .len = ESP_UUID_LEN_16,
//     .uuid = {.uuid16 = REMOTE_SERVICE_UUID,},
// };

// static esp_bt_uuid_t remote_filter_char_uuid = {
//     .len = ESP_UUID_LEN_16,
//     .uuid = {.uuid16 = REMOTE_NOTIFY_CHAR_UUID,},
// };

// static esp_bt_uuid_t notify_descr_uuid = {
//     .len = ESP_UUID_LEN_16,
//     .uuid = {.uuid16 = ESP_GATT_UUID_CHAR_CLIENT_CONFIG,},
// };

static esp_ble_scan_params_t ble_scan_params = {
    .scan_type              = BLE_SCAN_TYPE_ACTIVE,
    .own_addr_type          = BLE_ADDR_TYPE_PUBLIC,
    .scan_filter_policy     = BLE_SCAN_FILTER_ALLOW_ALL,
    .scan_interval          = 0x50,
    .scan_window            = 0x30,
    .scan_duplicate         = BLE_SCAN_DUPLICATE_DISABLE
};

struct gattc_profile_inst {
    esp_gattc_cb_t gattc_cb;
    uint16_t gattc_if;
    uint16_t app_id;
    uint16_t conn_id;
    uint16_t service_start_handle;
    uint16_t service_end_handle;
    uint16_t char_handle;
    esp_bd_addr_t remote_bda;
};

/* One gatt-based profile one app_id and one gattc_if, this array will store the gattc_if returned by ESP_GATTS_REG_EVT */
static struct gattc_profile_inst gl_profile_tab[PROFILE_NUM] = {
    [PROFILE_A_APP_ID] = {
        .gattc_cb = gattc_profile_event_handler,
        .gattc_if = ESP_GATT_IF_NONE,       /* Not get the gatt_if, so initial is ESP_GATT_IF_NONE */
    },
};

static void gattc_profile_event_handler(esp_gattc_cb_event_t event, esp_gatt_if_t gattc_if, esp_ble_gattc_cb_param_t *param)
{
    esp_ble_gattc_cb_param_t *p_data = (esp_ble_gattc_cb_param_t *)param;

    switch (event) {
    case ESP_GATTC_REG_EVT:
        ESP_LOGI(GATTC_TAG, "REG_EVT");
        esp_err_t scan_ret = esp_ble_gap_set_scan_params(&ble_scan_params);
        if (scan_ret){
            ESP_LOGE(GATTC_TAG, "set scan params error, error code = %x", scan_ret);
        }
        break;
    case ESP_GATTC_CONNECT_EVT:{
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_CONNECT_EVT conn_id %d, if %d", p_data->connect.conn_id, gattc_if);
        gl_profile_tab[PROFILE_A_APP_ID].conn_id = p_data->connect.conn_id;
        memcpy(gl_profile_tab[PROFILE_A_APP_ID].remote_bda, p_data->connect.remote_bda, sizeof(esp_bd_addr_t));
        ESP_LOGI(GATTC_TAG, "REMOTE BDA:");
        esp_log_buffer_hex(GATTC_TAG, gl_profile_tab[PROFILE_A_APP_ID].remote_bda, sizeof(esp_bd_addr_t));
        esp_err_t mtu_ret = esp_ble_gattc_send_mtu_req (gattc_if, p_data->connect.conn_id);
        if (mtu_ret){
            ESP_LOGE(GATTC_TAG, "config MTU error, error code = %x", mtu_ret);
        }
        break;
    }
    case ESP_GATTC_OPEN_EVT:
        if (param->open.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "open failed, status %d", p_data->open.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "open success");
        break;
    case ESP_GATTC_DIS_SRVC_CMPL_EVT:
        if (param->dis_srvc_cmpl.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "discover service failed, status %d", param->dis_srvc_cmpl.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "discover service complete conn_id %d", param->dis_srvc_cmpl.conn_id);
        esp_ble_gattc_search_service(gattc_if, param->cfg_mtu.conn_id, &remote_filter_service_uuid);
        break;
    case ESP_GATTC_CFG_MTU_EVT:
        if (param->cfg_mtu.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG,"config mtu failed, error status = %x", param->cfg_mtu.status);
        }
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_CFG_MTU_EVT, Status %d, MTU %d, conn_id %d", param->cfg_mtu.status, param->cfg_mtu.mtu, param->cfg_mtu.conn_id);
        break;
    // case ESP_GATTC_SEARCH_RES_EVT: {
    //     ESP_LOGI(GATTC_TAG, "SEARCH RES: conn_id = %x is primary service %d", p_data->search_res.conn_id, p_data->search_res.is_primary);
    //     ESP_LOGI(GATTC_TAG, "start handle %d end handle %d current handle value %d", p_data->search_res.start_handle, p_data->search_res.end_handle, p_data->search_res.srvc_id.inst_id);
    //     if (p_data->search_res.srvc_id.uuid.len == ESP_UUID_LEN_16 && p_data->search_res.srvc_id.uuid.uuid.uuid16 == REMOTE_SERVICE_UUID) {
    //         ESP_LOGI(GATTC_TAG, "service found");
    //         get_server = true;
    //         gl_profile_tab[PROFILE_A_APP_ID].service_start_handle = p_data->search_res.start_handle;
    //         gl_profile_tab[PROFILE_A_APP_ID].service_end_handle = p_data->search_res.end_handle;
    //         ESP_LOGI(GATTC_TAG, "UUID16: %x", p_data->search_res.srvc_id.uuid.uuid.uuid16);
    //     }
    //     break;
    // }
    case ESP_GATTC_SEARCH_RES_EVT: {
        ESP_LOGI(GATTC_TAG, "SEARCH RES: conn_id = %x is primary service %d", p_data->search_res.conn_id, p_data->search_res.is_primary);
        ESP_LOGI(GATTC_TAG, "start handle %d end handle %d current handle value %d", p_data->search_res.start_handle, p_data->search_res.end_handle, p_data->search_res.srvc_id.inst_id);
        if (p_data->search_res.srvc_id.uuid.len == ESP_UUID_LEN_128) {
            if(memcmp(p_data->search_res.srvc_id.uuid.uuid.uuid128, gatts_service_uuid128, ESP_UUID_LEN_128) == 0){
                ESP_LOGI(GATTC_TAG, "service uuid128 found");
                get_server = true;
                gl_profile_tab[PROFILE_A_APP_ID].service_start_handle = p_data->search_res.start_handle;
                gl_profile_tab[PROFILE_A_APP_ID].service_end_handle = p_data->search_res.end_handle;
            } else {
                ESP_LOGE(GATTC_TAG, "service not found");
            }
        }
        break;
    }
    case ESP_GATTC_SEARCH_CMPL_EVT:
        if (p_data->search_cmpl.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "search service failed, error status = %x", p_data->search_cmpl.status);
            break;
        }
        if(p_data->search_cmpl.searched_service_source == ESP_GATT_SERVICE_FROM_REMOTE_DEVICE) {
            ESP_LOGI(GATTC_TAG, "Get service information from remote device");
        } else if (p_data->search_cmpl.searched_service_source == ESP_GATT_SERVICE_FROM_NVS_FLASH) {
            ESP_LOGI(GATTC_TAG, "Get service information from flash");
        } else {
            ESP_LOGI(GATTC_TAG, "unknown service source");
        }
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_SEARCH_CMPL_EVT");
        if (get_server){
            uint16_t count = 0;
            esp_gatt_status_t status = esp_ble_gattc_get_attr_count( gattc_if,
                                                                     p_data->search_cmpl.conn_id,
                                                                     ESP_GATT_DB_CHARACTERISTIC,
                                                                     gl_profile_tab[PROFILE_A_APP_ID].service_start_handle,
                                                                     gl_profile_tab[PROFILE_A_APP_ID].service_end_handle,
                                                                     INVALID_HANDLE,
                                                                     &count);
            if (status != ESP_GATT_OK){
                ESP_LOGE(GATTC_TAG, "esp_ble_gattc_get_attr_count error");
            }

            if (count > 0){
                char_elem_result = (esp_gattc_char_elem_t *)malloc(sizeof(esp_gattc_char_elem_t) * count);
                if (!char_elem_result){
                    ESP_LOGE(GATTC_TAG, "gattc no mem");
                }else{
                    status = esp_ble_gattc_get_char_by_uuid( gattc_if,
                                                             p_data->search_cmpl.conn_id,
                                                             gl_profile_tab[PROFILE_A_APP_ID].service_start_handle,
                                                             gl_profile_tab[PROFILE_A_APP_ID].service_end_handle,
                                                             remote_filter_char_uuid,
                                                             char_elem_result,
                                                             &count);
                    if (status != ESP_GATT_OK){
                        ESP_LOGE(GATTC_TAG, "esp_ble_gattc_get_char_by_uuid error");
                    }

                    /*  Every service have only one char in our 'ESP_GATTS_DEMO' demo, so we used first 'char_elem_result' */
                    if (count > 0 && (char_elem_result[0].properties & ESP_GATT_CHAR_PROP_BIT_NOTIFY)){
                        gl_profile_tab[PROFILE_A_APP_ID].char_handle = char_elem_result[0].char_handle;
                        esp_ble_gattc_register_for_notify (gattc_if, gl_profile_tab[PROFILE_A_APP_ID].remote_bda, char_elem_result[0].char_handle);
                    }
                }
                /* free char_elem_result */
                free(char_elem_result);
            }else{
                ESP_LOGE(GATTC_TAG, "no char found");
            }
        }
         break;
    case ESP_GATTC_REG_FOR_NOTIFY_EVT: {
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_REG_FOR_NOTIFY_EVT");
        if (p_data->reg_for_notify.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "REG FOR NOTIFY failed: error status = %d", p_data->reg_for_notify.status);
        }else{
            uint16_t count = 0;
            uint16_t notify_en = 1;
            esp_gatt_status_t ret_status = esp_ble_gattc_get_attr_count( gattc_if,
                                                                         gl_profile_tab[PROFILE_A_APP_ID].conn_id,
                                                                         ESP_GATT_DB_DESCRIPTOR,
                                                                         gl_profile_tab[PROFILE_A_APP_ID].service_start_handle,
                                                                         gl_profile_tab[PROFILE_A_APP_ID].service_end_handle,
                                                                         gl_profile_tab[PROFILE_A_APP_ID].char_handle,
                                                                         &count);
            if (ret_status != ESP_GATT_OK){
                ESP_LOGE(GATTC_TAG, "esp_ble_gattc_get_attr_count error");
            }
            if (count > 0){
                descr_elem_result = malloc(sizeof(esp_gattc_descr_elem_t) * count);
                if (!descr_elem_result){
                    ESP_LOGE(GATTC_TAG, "malloc error, gattc no mem");
                }else{
                    ret_status = esp_ble_gattc_get_descr_by_char_handle( gattc_if,
                                                                         gl_profile_tab[PROFILE_A_APP_ID].conn_id,
                                                                         p_data->reg_for_notify.handle,
                                                                         notify_descr_uuid,
                                                                         descr_elem_result,
                                                                         &count);
                    if (ret_status != ESP_GATT_OK){
                        ESP_LOGE(GATTC_TAG, "esp_ble_gattc_get_descr_by_char_handle error");
                    }
                    /* Every char has only one descriptor in our 'ESP_GATTS_DEMO' demo, so we used first 'descr_elem_result' */
                    if (count > 0 && descr_elem_result[0].uuid.len == ESP_UUID_LEN_16 && descr_elem_result[0].uuid.uuid.uuid16 == ESP_GATT_UUID_CHAR_CLIENT_CONFIG){
                        ret_status = esp_ble_gattc_write_char_descr( gattc_if,
                                                                     gl_profile_tab[PROFILE_A_APP_ID].conn_id,
                                                                     descr_elem_result[0].handle,
                                                                     sizeof(notify_en),
                                                                     (uint8_t *)&notify_en,
                                                                     ESP_GATT_WRITE_TYPE_RSP,
                                                                     ESP_GATT_AUTH_REQ_NONE);
                    }

                    if (ret_status != ESP_GATT_OK){
                        ESP_LOGE(GATTC_TAG, "esp_ble_gattc_write_char_descr error");
                    }

                    /* free descr_elem_result */
                    free(descr_elem_result);
                }
            }
            else{
                ESP_LOGE(GATTC_TAG, "decsr not found");
            }

        }
        break;
    }
    case ESP_GATTC_NOTIFY_EVT:
        if (p_data->notify.is_notify){
            ESP_LOGI(GATTC_TAG, "ESP_GATTC_NOTIFY_EVT, receive notify value:");
        }else{
            ESP_LOGI(GATTC_TAG, "ESP_GATTC_NOTIFY_EVT, receive indicate value:");
        }
        
        // esp_log_buffer_hex(GATTC_TAG, p_data->notify.value, p_data->notify.value_len);
        usleep(100000);
        for (int i = 0; i < BYTES_PER_LINE/sizeof(float)/3; i ++) {
            float *x = (float*)&p_data->notify.value[3 * i * sizeof(float)];
            float *y = (float*)&p_data->notify.value[3 * i * sizeof(float) + sizeof(float)];
            float *z = (float*)&p_data->notify.value[3 * i * sizeof(float) + sizeof(float) * 2];
            // printf("x: %f, y: %f, z: %f\n", *x, *y, *z); 
            // double total = sqrt(pow(*x, 2) + pow(*y, 2) + pow(*z, 2));
            sensor_x_data[i] = *x;
            sensor_y_data[i] = *y;
            sensor_z_data[i] = *z;
            // sensor_tol_data[i] = total;       
        }


        // peak detection
        for (int i = 0; i < num; ++i)
        {
            if(isnan(sensor_tol_data[i]) || sensor_tol_data[i] > 4000)
                flag = 0;
        }
        if(flag)
        {
            if (first == 0)
            {
                first = 1;
                printf("Start detection\n");
            }
            for (int i = 0; i < num; ++i)
            {
                // printf("Sensor %d: x: %f, y: %f, z: %f, total: %f\n", i + 1, sensor_x_data[i], sensor_y_data[i], sensor_z_data[i], sensor_tol_data[i]);
                double temp_x = sensor_x_data[i] - start_raw_amp_x[i];
                double temp_y = sensor_y_data[i] - start_raw_amp_y[i];
                double temp_z = sensor_z_data[i] - start_raw_amp_z[i];
                // printf("Sensor %d: x: %f, y: %f, z: %f, total: %f\n", i + 1, sensor_x_data[i], sensor_y_data[i], sensor_z_data[i], sensor_tol_data[i]);
                sensor_tol_data[i] = sqrt(pow(temp_x, 2) + pow(temp_y, 2) + pow(temp_z, 2));
                appendToDynamicArray(&sensors[i], temp_x, temp_y, temp_z, sensor_tol_data[i]);
                // appendToDynamicArray(&sensors[i], sensor_x_data[i], sensor_y_data[i], sensor_z_data[i], sensor_tol_data[i]);
            }

            // 1. smooth the data
            if (cnt < WND)
            {
                for(int i = 0; i < num; ++i)
                {
                    double x = sensors[i].data[cnt].x - start_raw_amp_x[i];
                    double y = sensors[i].data[cnt].y - start_raw_amp_y[i];
                    double z = sensors[i].data[cnt].z - start_raw_amp_z[i];

                    double tol = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
                    appendToDynamicArray(&smooth_sensors[i], x, y, z, tol);
                    appendToDynamicArray(&derivative_sensors[i], 0.0, 0.0, 0.0, 0.0);
                    appendToDynamicArray(&smooth_derivative_sensors[i], 0.0, 0.0, 0.0, 0.0);
                }
            }
            else
            {
                // 1. smooth the raw data
                for(int i = 0; i < num; i++)
                {
                    double sum_x = 0.0;
                    double sum_y = 0.0;
                    double sum_z = 0.0;
                    double sum_tol = 0.0;
                    for (int j = 0; j < WND; j++) {
                        sum_x += SmoothVector_r[j] * sensors[i].data[cnt - WND + j].x;
                        sum_y += SmoothVector_r[j] * sensors[i].data[cnt - WND + j].y;
                        sum_z += SmoothVector_r[j] * sensors[i].data[cnt - WND + j].z;
                        sum_tol += SmoothVector_r[j] * sensors[i].data[cnt - WND + j].total;
                    }
                    // printf("sum_tol: %f\n", sum_tol);
                    appendToDynamicArray(&smooth_sensors[i], sum_x, sum_y, sum_z, sum_tol);
                }

                // 2. calculate the derivative
                for(int i = 0; i < num; i++)
                {
                    double last_point = smooth_sensors[i].data[smooth_sensors[i].size-1].total;
                    double first_point = smooth_sensors[i].data[smooth_sensors[i].size-2].total;
                    // printf("last_point: %f\n", last_point);
                    // printf("first_point: %f\n", first_point);
                    double derivative = (last_point - first_point) / 2.0;
                    // printf("derivative: %f\n", derivative);
                    appendToDynamicArray(&derivative_sensors[i], 0.0, 0.0, 0.0, derivative);
                }

                // 3. smooth the derivative
                for(int i = 0; i < num; i++)
                {
                    double sum_x = 0.0;
                    double sum_y = 0.0;
                    double sum_z = 0.0;
                    double sum_tol = 0.0;
                    for (int j = 0; j < WND_D; j++) {
                        // sum_x += SmoothVector_d[j] * derivative_sensors[i].data[cnt - WND_D + j].x;
                        // sum_y += SmoothVector_d[j] * derivative_sensors[i].data[cnt - WND_D + j].y;
                        // sum_z += SmoothVector_d[j] * derivative_sensors[i].data[cnt - WND_D + j].z;
                        sum_tol += SmoothVector_d[j] * derivative_sensors[i].data[cnt - WND_D + j].total;
                    }
                    appendToDynamicArray(&smooth_derivative_sensors[i], sum_x, sum_y, sum_z, sum_tol);
                }

                // 4. peak detection
                for(int i = 0; i < num; i++)
                {
                    S_flag[i] = 0;

                    if(slope_list[i].length == 10)
                    {
                        if(AuxiliaryFlag[i])
                        {
                            double tmp = 0.0;
                            for(int j = 1; j < slope_list[i].length; j++)
                                tmp += fabs(slope_list[i].data[j]);
                            slope_thr[i] = amp_thrd[i] * (tmp / (slope_list[i].length - 1));
                            AuxiliaryFlag[i] = false;
                            for(int j = 1; j < sensors[i].size; j++)
                            {
                                start_raw_amp_x[i] += sensors[i].data[j].x;
                                start_raw_amp_y[i] += sensors[i].data[j].y;
                                start_raw_amp_z[i] += sensors[i].data[j].z;
                            }
                            start_raw_amp_x[i] /= sensors[i].size;
                            start_raw_amp_y[i] /= sensors[i].size;
                            start_raw_amp_z[i] /= sensors[i].size;
                            printf("Sensor %d: slope_thr = %f\n", i + 1, slope_thr[i]);
                        }
                    }
                    
                    // Positive peak
                    double smoothed_1 = smooth_derivative_sensors[i].data[smooth_derivative_sensors[i].size-1].total;
                    double smoothed_2 = smooth_derivative_sensors[i].data[smooth_derivative_sensors[i].size-2].total;
                    if(smoothed_1 <= 0.0 && smoothed_2 >= 0.0)
                    {
                        double slope = smoothed_1 - smoothed_2;
                        slope_list[i].length++;
                        slope_list[i].data = (double *)realloc(slope_list[i].data, slope_list[i].length * sizeof(double));
                        if (slope_list[i].data == NULL) {
                            //
                            fprintf(stderr, "Memory allocation failed.\n");
                            // return 1;
                        }
                        slope_list[i].data[slope_list[i].length - 1] = slope;
                        // printf("%d", slope_list[i].length);
                        int raw_index = cnt - WND;
                        double raw = smooth_sensors[i].data[raw_index].total;
            
                        if (slope <= -slope_thr[i])
                        {
                            // if(fabs(LastRAW[i]) <= 1e-6)
                            // {
                            //     S_flag[i] = 1;
                            //     no = no + 1;
                            //     // printf("%d\n", raw_index);
                            //     // printf("%f\n", slope);
                            //     printf("Sensor %d: No. %d detect a magnet\n", i + 1, no);
                            //     // printf("Sensor %d: raw data: %f\n", i + 1, raw);
                            //         // print x, y, z data
                            //     printf("Sensor %d: x: %f, y: %f, z: %f\n", i + 1, smooth_sensors[i].data[raw_index].x, smooth_sensors[i].data[raw_index].y, smooth_sensors[i].data[raw_index].z);

                            // }
                            if(slope_list[i].length > 10)
                            {
                                if(raw >= delta_thrd[i])
                                {
                                    amp_tol = raw;
                                    double amp_tmp_x = smooth_sensors[i].data[raw_index].x;
                                    double amp_tmp_y = smooth_sensors[i].data[raw_index].y;
                                    double amp_tmp_z = smooth_sensors[i].data[raw_index].z;
                                    amp_tmp_list[i].x = amp_tmp_x;
                                    amp_tmp_list[i].y = amp_tmp_y;
                                    amp_tmp_list[i].z = amp_tmp_z;
                                    amp_tmp_list[i].total = amp_tol;
                                    S_flag[i] = 1;
                                    cur_cnt = cnt;
                                    flag_mag_det = true;
                                    no = no + 1;
                                    // printf("%d\n", raw_index);
                                    // printf("%f\n", slope);
                                    printf("Sensor %d: No. %d detect a magnet\n", i + 1, no);
                                    // print the raw data
                                    // printf("Sensor %d: raw data: %f\n", i + 1, raw);
                                    // print x, y, z data
                                    printf("Sensor %d: x: %f, y: %f, z: %f\n", i + 1, smooth_sensors[i].data[raw_index].x, smooth_sensors[i].data[raw_index].y, smooth_sensors[i].data[raw_index].z);
                                } 
                            }
                        
                        }
                        else
                        {
                            int forward_points = 20;
                            // for(int i = 0; i < num; i++)
                            // {
                            //     // LastRAW[i] = (np.array(sz[i][raw_index - 15: raw_index-10])).mean()
                            //     double s = 0.0;
                            //     int interval = 15;
                            //     for (int j = 0; j < interval; j++)
                            //         s += smooth_sensors[i].data[raw_index-40+j].total;
                            //     LastRAW[i] = s / 15.0;
                            // }
                        }     
                    }
                }

                // 

                magnetInfo.cur_n = 0;
                magnetInfo.index_flag[0] = 0;
                magnetInfo.index_flag[1] = 0;
                magnetInfo.index_flag[2] = 0;
                magnetInfo.tmp_data[0].x = 0.0;
                magnetInfo.tmp_data[0].y = 0.0;
                magnetInfo.tmp_data[0].z = 0.0;
                magnetInfo.tmp_data[0].total = 0.0;
                magnetInfo.tmp_data[1].x = 0.0;
                magnetInfo.tmp_data[1].y = 0.0;
                magnetInfo.tmp_data[1].z = 0.0;
                magnetInfo.tmp_data[1].total = 0.0;
                magnetInfo.tmp_data[2].x = 0.0;
                magnetInfo.tmp_data[2].y = 0.0;
                magnetInfo.tmp_data[2].z = 0.0;
                magnetInfo.tmp_data[2].total = 0.0;

                if(flag_mag_det)
                {
                    //magnet_info
                    magnetInfo.cur_n = cur_cnt;
                    magnetInfo.index_flag[0] = S_flag[0];
                    magnetInfo.index_flag[1] = S_flag[1];
                    magnetInfo.index_flag[2] = S_flag[2];
                    magnetInfo.tmp_data[0].x = amp_tmp_list[0].x;
                    magnetInfo.tmp_data[0].y = amp_tmp_list[0].y;
                    magnetInfo.tmp_data[0].z = amp_tmp_list[0].z;
                    magnetInfo.tmp_data[0].total = amp_tmp_list[0].total;
                    magnetInfo.tmp_data[1].x = amp_tmp_list[1].x;
                    magnetInfo.tmp_data[1].y = amp_tmp_list[1].y;
                    magnetInfo.tmp_data[1].z = amp_tmp_list[1].z;
                    magnetInfo.tmp_data[1].total = amp_tmp_list[1].total;
                    magnetInfo.tmp_data[2].x = amp_tmp_list[2].x;
                    magnetInfo.tmp_data[2].y = amp_tmp_list[2].y;
                    magnetInfo.tmp_data[2].z = amp_tmp_list[2].z;
                    magnetInfo.tmp_data[2].total = amp_tmp_list[2].total;
                    flag_mag_det = false;
                }

                if(magnetInfo.cur_n > 0)
                {
                    int tmp_cnt = magnetInfo.cur_n;
                    int *tmp_ind = magnetInfo.index_flag;
                    for(int jj = 0; jj < 3; jj++)
                    {
                        // sensor i detect a magnet
                        if(tmp_ind[jj] == 1)
                        {

                            int after_ind = 10;
                            // double amp_x_ddtw[30];
                            // double amp_y_ddtw[30];
                            // double amp_z_ddtw[30];
                            
                            for(int kk = 0; kk < 20; kk++)
                            {
                                // amp_x_ddtw[kk] = 
                                // amp_y_ddtw[kk] = smooth_sensors[jj].data[tmp_cnt-after_ind + kk].y;
                                // amp_z_ddtw[kk] = smooth_sensors[jj].data[tmp_cnt-after_ind + kk].z;
                                amp_tol_ddtw[kk][0] = smooth_sensors[jj].data[tmp_cnt-after_ind + kk].x;
                                amp_tol_ddtw[kk][1] = smooth_sensors[jj].data[tmp_cnt-after_ind + kk].y;
                                amp_tol_ddtw[kk][2] = smooth_sensors[jj].data[tmp_cnt-after_ind + kk].z;
                            }
                            int ang_gran = 8;
                            int dis_gran = 1;
                            int test_points = 20;
                            int gt_points = 20;
                            int axis = 3;
                            double angle = 0;
                            double start_time = clock();
                            angle = DDTW_matching_res(amp_tol_ddtw, ang_gran, dis_gran, test_points, gt_points, axis);
                            printf("Estimated angle is: %lf", angle);
                            double end_time = clock();
                            printf("DDTW time: %lf", (end_time - start_time) / CLOCKS_PER_SEC);

                        }
                    }
                    double amp1_localize_x = magnetInfo.tmp_data[0].x;
                    double amp1_localize_y = magnetInfo.tmp_data[0].y;
                    double amp1_localize_z = magnetInfo.tmp_data[0].z;
                    double amp1_localize[3] = {amp1_localize_x, amp1_localize_y, amp1_localize_z};

                    double amp3_localize_x = magnetInfo.tmp_data[2].x;
                    double amp3_localize_y = magnetInfo.tmp_data[2].y;
                    double amp3_localize_z = magnetInfo.tmp_data[2].z;
                    double amp3_localize[3] = {amp3_localize_x, amp3_localize_y, amp3_localize_z};
                    double sol[1] = {0};
                    double start_time = clock();
                    ComputeD(amp1_localize, amp3_localize, 0, 1, 3, 10, 8, sol);
                    double end_time = clock();
                    printf("Localize time: %lf", (end_time - start_time) / CLOCKS_PER_SEC);
                    printf("Localize result: %lf", sol[0]);
                    // freeMagnetInfo(magnetInfo);
                    // initMagnetInfo(magnetInfo, 5);
                }

            }
            cnt ++;
        }


        break;
    case ESP_GATTC_WRITE_DESCR_EVT:
        if (p_data->write.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "write descr failed, error status = %x", p_data->write.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "write descr success ");
        uint8_t write_char_data[35];
        for (int i = 0; i < sizeof(write_char_data); ++i)
        {
            write_char_data[i] = i % 256;
        }
        esp_ble_gattc_write_char( gattc_if,
                                  gl_profile_tab[PROFILE_A_APP_ID].conn_id,
                                  gl_profile_tab[PROFILE_A_APP_ID].char_handle,
                                  sizeof(write_char_data),
                                  write_char_data,
                                  ESP_GATT_WRITE_TYPE_RSP,
                                  ESP_GATT_AUTH_REQ_NONE);
        break;
    case ESP_GATTC_SRVC_CHG_EVT: {
        esp_bd_addr_t bda;
        memcpy(bda, p_data->srvc_chg.remote_bda, sizeof(esp_bd_addr_t));
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_SRVC_CHG_EVT, bd_addr:");
        esp_log_buffer_hex(GATTC_TAG, bda, sizeof(esp_bd_addr_t));
        break;
    }
    case ESP_GATTC_WRITE_CHAR_EVT:
        if (p_data->write.status != ESP_GATT_OK){
            ESP_LOGE(GATTC_TAG, "write char failed, error status = %x", p_data->write.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "write char success ");
        break;
    case ESP_GATTC_DISCONNECT_EVT:
        connect = false;
        get_server = false;
        ESP_LOGI(GATTC_TAG, "ESP_GATTC_DISCONNECT_EVT, reason = %d", p_data->disconnect.reason);
        break;
    default:
        break;
    }
}

static void esp_gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param)
{
    uint8_t *adv_name = NULL;
    uint8_t adv_name_len = 0;
    switch (event) {
    case ESP_GAP_BLE_SCAN_PARAM_SET_COMPLETE_EVT: {
        //the unit of the duration is second
        uint32_t duration = 30;
        esp_ble_gap_start_scanning(duration);
        break;
    }
    case ESP_GAP_BLE_SCAN_START_COMPLETE_EVT:
        //scan start complete event to indicate scan start successfully or failed
        if (param->scan_start_cmpl.status != ESP_BT_STATUS_SUCCESS) {
            ESP_LOGE(GATTC_TAG, "scan start failed, error status = %x", param->scan_start_cmpl.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "scan start success");

        break;
    case ESP_GAP_BLE_SCAN_RESULT_EVT: {
        esp_ble_gap_cb_param_t *scan_result = (esp_ble_gap_cb_param_t *)param;
        switch (scan_result->scan_rst.search_evt) {
        case ESP_GAP_SEARCH_INQ_RES_EVT:
            esp_log_buffer_hex(GATTC_TAG, scan_result->scan_rst.bda, 6);
            ESP_LOGI(GATTC_TAG, "searched Adv Data Len %d, Scan Response Len %d", scan_result->scan_rst.adv_data_len, scan_result->scan_rst.scan_rsp_len);
            adv_name = esp_ble_resolve_adv_data(scan_result->scan_rst.ble_adv,
                                                ESP_BLE_AD_TYPE_NAME_CMPL, &adv_name_len);
            ESP_LOGI(GATTC_TAG, "searched Device Name Len %d", adv_name_len);
            esp_log_buffer_char(GATTC_TAG, adv_name, adv_name_len);

#if CONFIG_EXAMPLE_DUMP_ADV_DATA_AND_SCAN_RESP
            if (scan_result->scan_rst.adv_data_len > 0) {
                ESP_LOGI(GATTC_TAG, "adv data:");
                esp_log_buffer_hex(GATTC_TAG, &scan_result->scan_rst.ble_adv[0], scan_result->scan_rst.adv_data_len);
            }
            if (scan_result->scan_rst.scan_rsp_len > 0) {
                ESP_LOGI(GATTC_TAG, "scan resp:");
                esp_log_buffer_hex(GATTC_TAG, &scan_result->scan_rst.ble_adv[scan_result->scan_rst.adv_data_len], scan_result->scan_rst.scan_rsp_len);
            }
#endif
            ESP_LOGI(GATTC_TAG, "\n");

            if (adv_name != NULL) {
                if (strlen(remote_device_name) == adv_name_len && strncmp((char *)adv_name, remote_device_name, adv_name_len) == 0) {
                    ESP_LOGI(GATTC_TAG, "searched device %s\n", remote_device_name);
                    if (connect == false) {
                        connect = true;
                        ESP_LOGI(GATTC_TAG, "connect to the remote device.");
                        esp_ble_gap_stop_scanning();
                        esp_ble_gattc_open(gl_profile_tab[PROFILE_A_APP_ID].gattc_if, scan_result->scan_rst.bda, scan_result->scan_rst.ble_addr_type, true);
                    }
                }
            }
            break;
        case ESP_GAP_SEARCH_INQ_CMPL_EVT:
            break;
        default:
            break;
        }
        break;
    }

    case ESP_GAP_BLE_SCAN_STOP_COMPLETE_EVT:
        if (param->scan_stop_cmpl.status != ESP_BT_STATUS_SUCCESS){
            ESP_LOGE(GATTC_TAG, "scan stop failed, error status = %x", param->scan_stop_cmpl.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "stop scan successfully");
        break;

    case ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT:
        if (param->adv_stop_cmpl.status != ESP_BT_STATUS_SUCCESS){
            ESP_LOGE(GATTC_TAG, "adv stop failed, error status = %x", param->adv_stop_cmpl.status);
            break;
        }
        ESP_LOGI(GATTC_TAG, "stop adv successfully");
        break;
    case ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT:
         ESP_LOGI(GATTC_TAG, "update connection params status = %d, min_int = %d, max_int = %d,conn_int = %d,latency = %d, timeout = %d",
                  param->update_conn_params.status,
                  param->update_conn_params.min_int,
                  param->update_conn_params.max_int,
                  param->update_conn_params.conn_int,
                  param->update_conn_params.latency,
                  param->update_conn_params.timeout);
        break;
    default:
        break;
    }
}

static void esp_gattc_cb(esp_gattc_cb_event_t event, esp_gatt_if_t gattc_if, esp_ble_gattc_cb_param_t *param)
{
    /* If event is register event, store the gattc_if for each profile */
    if (event == ESP_GATTC_REG_EVT) {
        if (param->reg.status == ESP_GATT_OK) {
            gl_profile_tab[param->reg.app_id].gattc_if = gattc_if;
        } else {
            ESP_LOGI(GATTC_TAG, "reg app failed, app_id %04x, status %d",
                    param->reg.app_id,
                    param->reg.status);
            return;
        }
    }

    /* If the gattc_if equal to profile A, call profile A cb handler,
     * so here call each profile's callback */
    do {
        int idx;
        for (idx = 0; idx < PROFILE_NUM; idx++) {
            if (gattc_if == ESP_GATT_IF_NONE || /* ESP_GATT_IF_NONE, not specify a certain gatt_if, need to call every profile cb function */
                    gattc_if == gl_profile_tab[idx].gattc_if) {
                if (gl_profile_tab[idx].gattc_cb) {
                    gl_profile_tab[idx].gattc_cb(event, gattc_if, param);
                }
            }
        }
    } while (0);
}



void app_main(void)
{   
   
    for (int i = 0; i < WND; i++) 
    {
        double x_val = i + 1;
        SmoothVector_r[i] = gaussian(x_val, WND / 2.0, WND / 2.0);
    }
    normalize(SmoothVector_r, WND);

    for (int i = 0; i < WND_D; i++) 
    {
        double x_val = i + 1;
        SmoothVector_d[i] = gaussian(x_val, WND_D / 2.0, WND_D / 2.0);
    }
    normalize(SmoothVector_d, WND_D);

    for (int i = 0; i < num; ++i)
        slope_thr[i] = 10000;

    for (int i = 0; i < num; ++i)
        LastRAW[i] = 0.0;

    for (int i = 0; i < num; ++i)
        S_flag[i] = 0;

    for (int i = 0; i < num; ++i)
        delta_thrd[i] = 20;

    for (int i = 0; i < num; ++i)
        amp_thrd[i] = 1.2;

    for (int i = 0; i < num; ++i)
        AuxiliaryFlag[i] = true;

    for (int i = 0; i < num; ++i)
    {
        start_raw_amp_x[i] = 0.0;
        start_raw_amp_y[i] = 0.0;
        start_raw_amp_z[i] = 0.0;
    }

    for (int i = 0; i < num; ++i) 
    {
        slope_list[i].length = 0; // 初始化长度为0
        slope_list[i].data = NULL; // 初始化数据指针为NULL
    }
    for (int i = 0; i < num; ++i)
    {
        initDynamicArray(&sensors[i], 50); 
        initDynamicArray(&smooth_sensors[i], 50);
        initDynamicArray(&derivative_sensors[i], 50);
        initDynamicArray(&smooth_derivative_sensors[i], 50);
        amp_tmp_list[i].x = amp_tmp_list[i].y = amp_tmp_list[i].z = amp_tmp_list[i].total = 0;
    }


    // Initialize NVS.
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));
    
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ret = esp_bt_controller_init(&bt_cfg);
    if (ret) {
        ESP_LOGE(GATTC_TAG, "%s initialize controller failed: %s\n", __func__, esp_err_to_name(ret));
        return;
    }

    ret = esp_bt_controller_enable(ESP_BT_MODE_BLE);
    if (ret) {
        ESP_LOGE(GATTC_TAG, "%s enable controller failed: %s\n", __func__, esp_err_to_name(ret));
        return;
    }

    ret = esp_bluedroid_init();
    if (ret) {
        ESP_LOGE(GATTC_TAG, "%s init bluetooth failed: %s\n", __func__, esp_err_to_name(ret));
        return;
    }

    ret = esp_bluedroid_enable();
    if (ret) {
        ESP_LOGE(GATTC_TAG, "%s enable bluetooth failed: %s\n", __func__, esp_err_to_name(ret));
        return;
    }

    //register the  callback function to the gap module
    ret = esp_ble_gap_register_callback(esp_gap_cb);
    if (ret){
        ESP_LOGE(GATTC_TAG, "%s gap register failed, error code = %x\n", __func__, ret);
        return;
    }

    //register the callback function to the gattc module
    ret = esp_ble_gattc_register_callback(esp_gattc_cb);
    if(ret){
        ESP_LOGE(GATTC_TAG, "%s gattc register failed, error code = %x\n", __func__, ret);
        return;
    }

    ret = esp_ble_gattc_app_register(PROFILE_A_APP_ID);
    if (ret){
        ESP_LOGE(GATTC_TAG, "%s gattc app register failed, error code = %x\n", __func__, ret);
    }
    esp_err_t local_mtu_ret = esp_ble_gatt_set_local_mtu(500);
    if (local_mtu_ret){
        ESP_LOGE(GATTC_TAG, "set local  MTU failed, error code = %x", local_mtu_ret);
    }

}
