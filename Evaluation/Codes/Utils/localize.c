#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

const double pi = 3.14159265358979323846;
// Parameter we want to sovle
const int N = 1;


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
double ComputeD(double amp1[3], double amp2[3], double theta, int index1, int index2, double h, double l, double sol[N]);
// first kind elliptic integral
double ellipe(double k, int N);
// second kind elliptic integral
double ellipk(double k, int N);
double cel0(double kc, double p, double c, double s);
// fieldH_cylinder_diametral
void fieldH_cylinder_diametral(double z0, double r, double phi, double z, double res[3]);
// magnetic_field_cylinder
void magnetic_field_cylinder(const char *field, double observer[3], double magnetization[3], double dimension[3], double final_res[3]);



int main()
{
    // Test the magnetic_field_cylinder function
    double observer[3] = {-2.14, 0, 5};
    double observer2[3] = {17.86, 0, 5};
    double magnetization[3] = {0, 100000, 0};
    double dimension[2] = {4, 1};
    double l = 20;

    double res[3];
    magnetic_field_cylinder("B", observer, magnetization, dimension, res);
    double res2[3];
    magnetic_field_cylinder("B", observer2, magnetization, dimension, res2);
    printf("Bx: %lf, By: %lf, Bz: %lf\n", res[0], res[1], res[2]);
    printf("Bx_2: %lf, By_2: %lf, Bz_2: %lf\n", res2[0], res2[1], res2[2]);
    double sol[N];
    ComputeD(res, res2, 0, 0, 1, 5, 20, sol);
    printf("sol1: %lf, sol2: %lf\n", sol[0], l-sol[0]);
    return 0;
}


// Check if the input is a valid field input
int check_field_input(const char *inp, const char *origin) 
{
    if (strcmp(inp, "B") == 0) {
        return 1;  // True
    }
    if (strcmp(inp, "H") == 0) {
        return 0;  // False
    }
    fprintf(stderr, "%s input can only be `field='B'` or `field='H'`.\n"
                    "Instead received %s.\n", origin, inp);
    exit(EXIT_FAILURE);
}

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
    magnetic_field_cylinder("B", observer, magnetization, dimension, res);
}


// Function to calculate the objective function
double objective_function(double r[], double amp1[3], double amp2[3], double theta, double l, double h) 
{
    // r2 = l - r1;
    double r1 = r[0];
    double r2 = l-r1;
    double B1_res[3], B2_res[3];
    B_simplified(-r1 * cos(theta), r1 * sin(theta), h, B1_res);
    double B1x = B1_res[0], B1y = B1_res[1], B1z = B1_res[2];

    B_simplified(r2 * cos(theta), -r2 * sin(theta), h, B2_res);
    double B2x = B2_res[0], B2y = B2_res[1], B2z = B2_res[2];

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
    double simplex[N+1][N], f[N+1];
    double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
    double epsilon = 1e-8;  // Tolerance for convergence
    int iter, max_iter = 400;
    
    // Initialize simplex with initial guess x
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j < N; j++) {
            simplex[i][j] = x[j];
        }
        if (i > 0) {
            simplex[i][i-1] += 1.0;
        }
    }
    
    iter = 0;
    do {
        // Evaluate objective function for each vertex of the simplex
        for (int i = 0; i <= N; i++) {
            f[i] = objective_function(simplex[i], amp1, amp2, theta, l, h);
        }
        
        // Sort simplex vertices based on objective function values
        for (int i = 0; i <= N; i++) {
            for (int j = i+1; j <= N; j++) {
                if (f[j] < f[i]) {
                    double temp = f[i];
                    f[i] = f[j];
                    f[j] = temp;
                    for (int k = 0; k < N; k++) {
                        temp = simplex[i][k];
                        simplex[i][k] = simplex[j][k];
                        simplex[j][k] = temp;
                    }
                }
            }
        }
        
        // Check convergence
        double diff = fabs(f[N] - f[0]);
        if (diff < epsilon) {
            break;  // Convergence achieved
        }
        
        // Compute centroid of best N vertices
        double centroid[N];
        for (int i = 0; i < N; i++) {
            centroid[i] = 0.0;
            for (int j = 0; j < N; j++) {
                centroid[i] += simplex[j][i];
            }
            centroid[i] /= N;
        }
        
        // Reflection
        double xr[N];
        for (int i = 0; i < N; i++) {
            xr[i] = centroid[i] + alpha * (centroid[i] - simplex[N][i]);
        }
        if (constraint(xr, l)) {
            double fr = objective_function(xr, amp1, amp2, theta, l, h);
            if (fr < f[0]) {
                for (int i = 0; i < N; i++) {
                    simplex[N][i] = xr[i];
                }
                f[N] = fr;
                continue;
            }
        }
        
        // Expansion
        if (objective_function(xr, amp1, amp2, theta, l, h) < f[N-1]) {
            double xe[N];
            for (int i = 0; i < N; i++) {
                xe[i] = centroid[i] + gamma * (xr[i] - centroid[i]);
            }
            if (constraint(xe, l)) {
                double fe = objective_function(xe, amp1, amp2, theta, l, h);
                if (fe < f[N]) {
                    for (int i = 0; i < N; i++) {
                        simplex[N][i] = xe[i];
                    }
                    f[N] = fe;
                    continue;
                }
            }
        }
        
        // Contraction
        double xc[N];
        for (int i = 0; i < N; i++) {
            xc[i] = centroid[i] + rho * (simplex[N][i] - centroid[i]);
        }
        if (constraint(xc, l)) {
            double fc = objective_function(xc, amp1, amp2, theta, l, h);
            if (fc < f[N]) {
                for (int i = 0; i < N; i++) {
                    simplex[N][i] = xc[i];
                }
                f[N] = fc;
                continue;
            }
        }
        
        // Shrink
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j < N; j++) {
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
    for (int i = 0; i < N; i++) {
        x[i] = simplex[0][i];
    }
}


double ComputeD(double amp1[3], double amp2[3], double theta, int index1, int index2, double h, double l, double sol[N]) 
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
        if (i == 0 || i == N)
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
        if (i == 0 || i == N)
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


void magnetic_field_cylinder(const char *field, double observer[3], double magnetization[3], double dimension[3], double final_res[3])
{
    int bh = check_field_input(field, "magnet_cylinder_field()");
    // transform to Cy CS --------------------------------------------
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