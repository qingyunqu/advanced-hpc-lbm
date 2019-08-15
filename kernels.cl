#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int jj,
                            float w1, float w2,
                            int total)
{
  int ii = get_global_id(0);

  float speeds[NSPEEDS];
  for(int i = 0; i < NSPEEDS; i++){
    speeds[i] = cells[ii + jj*nx + i*total];
  }

  if (!obstacles[ii + jj* nx]
      && (speeds[3] - w1) > 0.f
      && (speeds[6] - w2) > 0.f
      && (speeds[7] - w2) > 0.f)
  {
    speeds[1] += w1;
    speeds[5] += w2;
    speeds[8] += w2;
    speeds[3] -= w1;
    speeds[6] -= w2;
    speeds[7] -= w2;

    for(int i = 0; i < NSPEEDS; i++){
      cells[ii + jj*nx + i*total] = speeds[i];
    }
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  int ii = get_global_id(0) % nx;
  int jj = get_global_id(0) / nx;
  int total = nx * ny;

  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  float tmp_speeds[NSPEEDS];
  tmp_speeds[0] = cells[ii + jj*nx + 0*total]; /* central cell, no movement */
  tmp_speeds[1] = cells[x_w + jj*nx + 1*total]; /* east */
  tmp_speeds[2] = cells[ii + y_s*nx + 2*total]; /* north */
  tmp_speeds[3] = cells[x_e + jj*nx + 3*total]; /* west */
  tmp_speeds[4] = cells[ii + y_n*nx + 4*total]; /* south */
  tmp_speeds[5] = cells[x_w + y_s*nx + 5*total]; /* north-east */
  tmp_speeds[6] = cells[x_e + y_s*nx + 6*total]; /* north-west */
  tmp_speeds[7] = cells[x_e + y_n*nx + 7*total]; /* south-west */
  tmp_speeds[8] = cells[x_w + y_n*nx + 8*total]; /* south-east */

  for(int i=0; i < NSPEEDS; i++){
    tmp_cells[ii + jj*nx + i*total] = tmp_speeds[i];
  }
}

kernel void collision(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, float omega,
                      float c_sq, float w0,
                      float w1, float w2, int total)
{
  int ii = get_global_id(0);

  // rebound
  if(obstacles[ii])
  {
    float speeds[NSPEEDS];
    float tmp_speeds[NSPEEDS];
    for(int i = 0; i < NSPEEDS; i++){
      tmp_speeds[i] = tmp_cells[ii + i*total];
    }

    speeds[1] = tmp_speeds[3];
    speeds[2] = tmp_speeds[4];
    speeds[3] = tmp_speeds[1];
    speeds[4] = tmp_speeds[2];
    speeds[5] = tmp_speeds[7];
    speeds[6] = tmp_speeds[8];
    speeds[7] = tmp_speeds[5];
    speeds[8] = tmp_speeds[6];

    for(int i = 0; i < NSPEEDS; i++){
      cells[ii + i*total] = speeds[i];
    }
  }
  // collision
  else
  {
    float tmp_speeds[NSPEEDS];
    for(int i = 0; i < NSPEEDS; i++){
        tmp_speeds[i] = tmp_cells[ii + i*total];
    }

    float local_density = 0.f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_speeds[kk];
    }
    float u_x = (tmp_speeds[1]
                 + tmp_speeds[5]
                 + tmp_speeds[8]
                 - (tmp_speeds[3]
                    + tmp_speeds[6]
                    + tmp_speeds[7]))
                / local_density;
    float u_y = (tmp_speeds[2]
                 + tmp_speeds[5]
                 + tmp_speeds[6]
                 - (tmp_speeds[4]
                    + tmp_speeds[7]
                    + tmp_speeds[8]))
                 / local_density;
    float u_sq = u_x * u_x + u_y * u_y;

    float u[NSPEEDS];
    u[1] = u_x;
    u[2] = u_y;
    u[3] = - u_x;
    u[4] = - u_y;
    u[5] = u_x + u_y;
    u[6] = - u_x + u_y;
    u[7] = - u_x - u_y;
    u[8] = u_x - u_y;

    float c_sq_1 = 1 / c_sq;
    float u_sq_c_sq_2 = u_sq / (2 * c_sq);
    float c_sq_2_x2_1 = 1 / (2 * c_sq * c_sq);
    float w1_local_density = w1 * local_density;
    float w2_local_density = w2 * local_density;
    float d_equ[NSPEEDS];
    d_equ[0] = w0 * local_density * (1.f - u_sq_c_sq_2);
    d_equ[1] = w1_local_density * (1.f + u[1] * c_sq_1
                                       + (u[1] * u[1]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[2] = w1_local_density * (1.f + u[2] * c_sq_1
                                       + (u[2] * u[2]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[3] = w1_local_density * (1.f + u[3] * c_sq_1
                                       + (u[3] * u[3]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[4] = w1_local_density * (1.f + u[4] * c_sq_1
                                       + (u[4] * u[4]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[5] = w2_local_density * (1.f + u[5] * c_sq_1
                                       + (u[5] * u[5]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[6] = w2_local_density * (1.f + u[6] * c_sq_1
                                       + (u[6] * u[6]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[7] = w2_local_density * (1.f + u[7] * c_sq_1
                                       + (u[7] * u[7]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);
    d_equ[8] = w2_local_density * (1.f + u[8] * c_sq_1
                                       + (u[8] * u[8]) * c_sq_2_x2_1
                                       - u_sq_c_sq_2);

    float speeds[NSPEEDS];
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
        speeds[kk] = tmp_speeds[kk] + omega * (d_equ[kk] - tmp_speeds[kk]);
    }

    for(int i = 0; i < NSPEEDS; i++){
        cells[ii + i*total] = speeds[i];
    }
  }
}

kernel void av_velocity(global float* cells,
                        global int* obstacles,
                        global float* av_t,
                        int nx,
                        local float* local_sum,
                        int total)
{
  float tot_u = 0.f;

  int ii = get_global_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);
  int group_id = get_group_id(0);

  local_sum[local_id * 2] = 0.f;
  local_sum[local_id * 2 + 1] = 0.f;

  if (!obstacles[ii])
  {
    float speeds[NSPEEDS];
    for(int i = 0; i < NSPEEDS; i++){
      speeds[i] = cells[ii + i*total];
    }

    float local_density = 0.f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += speeds[kk];
    }
    float u_x = (speeds[1]
                 + speeds[5]
                 + speeds[8]
                 - (speeds[3]
                    + speeds[6]
                    + speeds[7]))
                / local_density;
    float u_y = (speeds[2]
                 + speeds[5]
                 + speeds[6]
                 - (speeds[4]
                    + speeds[7]
                    + speeds[8]))
                / local_density;
    local_sum[local_id * 2] = sqrt((u_x * u_x) + (u_y * u_y));
    local_sum[local_id * 2 + 1] = 1.f;

    for(int i = 0; i < NSPEEDS; i++){
      cells[ii + i*total] = speeds[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if(local_id == 0){
    for(int jj = 0; jj < local_size; jj++)
    {
      av_t[2 * group_id] += local_sum[jj * 2];
      av_t[2 * group_id +1] += local_sum[jj * 2 + 1];
    }
  }
}
