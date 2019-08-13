#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int jj,
                            float w1, float w2)
{
  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  t_speed cell = cells[ii + jj*nx];
  if (!obstacles[ii + jj* nx]
      && (cell.speeds[3] - w1) > 0.f
      && (cell.speeds[6] - w2) > 0.f
      && (cell.speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cell.speeds[1] += w1;
    cell.speeds[5] += w2;
    cell.speeds[8] += w2;
    /* decrease 'west-side' densities */
    cell.speeds[3] -= w1;
    cell.speeds[6] -= w2;
    cell.speeds[7] -= w2;

    cells[ii + jj*nx] = cell;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
}

kernel void rebound(global t_speed* cells,
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    int nx)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  if (obstacles[ii + jj*nx])
  {
    t_speed cell;
    t_speed tmp_cell = tmp_cells[ii + jj*nx];
    cell.speeds[1] = tmp_cell.speeds[3];
    cell.speeds[2] = tmp_cell.speeds[4];
    cell.speeds[3] = tmp_cell.speeds[1];
    cell.speeds[4] = tmp_cell.speeds[2];
    cell.speeds[5] = tmp_cell.speeds[7];
    cell.speeds[6] = tmp_cell.speeds[8];
    cell.speeds[7] = tmp_cell.speeds[5];
    cell.speeds[8] = tmp_cell.speeds[6];

    cells[ii + jj*nx] = cell;
  }
}

kernel void collision(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, float omega,
                      float c_sq, float w0,
                      float w1, float w2)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  if (!obstacles[ii + jj*nx])
  {
    float local_density = 0.f;
    t_speed tmp_cell = tmp_cells[ii + jj*nx];
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cell.speeds[kk];
    }
    float u_x = (tmp_cell.speeds[1]
                 + tmp_cell.speeds[5]
                 + tmp_cell.speeds[8]
                 - (tmp_cell.speeds[3]
                    + tmp_cell.speeds[6]
                    + tmp_cell.speeds[7]))
                / local_density;
    float u_y = (tmp_cell.speeds[2]
                 + tmp_cell.speeds[5]
                 + tmp_cell.speeds[6]
                 - (tmp_cell.speeds[4]
                    + tmp_cell.speeds[7]
                    + tmp_cell.speeds[8]))
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

    float d_equ[NSPEEDS];
    d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

    t_speed cell;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
        cell.speeds[kk] = tmp_cell.speeds[kk]
                                         + omega
                                         * (d_equ[kk] - tmp_cell.speeds[kk]);
    }
    cells[ii + jj*nx] = cell;
  }
}

kernel void av_velocity(global t_speed* cells,
                        global int* obstacles,
                        global float* av_t,
                        int nx)
{
  float tot_u = 0.f;

  int ii = get_global_id(0);
  int jj = get_global_id(1);

  if (!obstacles[ii + jj*nx])
  {
    float local_density = 0.f;
    t_speed cell = cells[ii + jj*nx];
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += cell.speeds[kk];
    }
    float u_x = (cell.speeds[1]
                 + cell.speeds[5]
                 + cell.speeds[8]
                 - (cell.speeds[3]
                    + cell.speeds[6]
                    + cell.speeds[7]))
                / local_density;
    float u_y = (cell.speeds[2]
                 + cell.speeds[5]
                 + cell.speeds[6]
                 - (cell.speeds[4]
                    + cell.speeds[7]
                    + cell.speeds[8]))
                / local_density;
    av_t[ii + jj*nx] = sqrt((u_x * u_x) + (u_y * u_y));
  }
  else
  {
    av_t[ii + jj*nx] = -1.f;
  }
}
