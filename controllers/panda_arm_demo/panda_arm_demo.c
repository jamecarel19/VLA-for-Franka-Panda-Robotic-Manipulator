/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/position_sensor.h>
#include <webots/keyboard.h>
#include <webots/camera.h>

#define TIME_STEP 32

enum Joints {
  JOINT1,
  JOINT2,
  JOINT3,
  JOINT4,
  JOINT5,
  JOINT6,
  JOINT7,
  FINGER
};

static WbDeviceTag motors[8];
static WbDeviceTag sensors[7];
static WbDeviceTag wrist_camera;
static double joint_min[7];
static double joint_max[7];

static double clamp(double value, double min, double max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

typedef struct {
  double x;
  double y;
  double z;
} Vec3;

static void mat4_identity(double m[4][4]) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      m[r][c] = (r == c) ? 1.0 : 0.0;
}

static void mat4_mul(const double a[4][4], const double b[4][4], double out[4][4]) {
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out[r][c] = 0.0;
      for (int k = 0; k < 4; ++k)
        out[r][c] += a[r][k] * b[k][c];
    }
  }
}

static void dh_transform(double a, double alpha, double d, double theta, double out[4][4]) {
  double ct = cos(theta);
  double st = sin(theta);
  double ca = cos(alpha);
  double sa = sin(alpha);

  out[0][0] = ct;
  out[0][1] = -st * ca;
  out[0][2] = st * sa;
  out[0][3] = a * ct;

  out[1][0] = st;
  out[1][1] = ct * ca;
  out[1][2] = -ct * sa;
  out[1][3] = a * st;

  out[2][0] = 0.0;
  out[2][1] = sa;
  out[2][2] = ca;
  out[2][3] = d;

  out[3][0] = 0.0;
  out[3][1] = 0.0;
  out[3][2] = 0.0;
  out[3][3] = 1.0;
}

// Full FK returning 4x4 transform
static void fk_panda(const double q[7], double out[4][4]) {
  // Approximate Franka Panda DH parameters.
  const double a[7] = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
  const double d[7] = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107};
  const double alpha[7] = {-M_PI_2, M_PI_2, M_PI_2, M_PI_2, -M_PI_2, M_PI_2, 0.0};

  mat4_identity(out);

  for (int i = 0; i < 7; ++i) {
    double a_i[4][4];
    double t_next[4][4];
    dh_transform(a[i], alpha[i], d[i], q[i], a_i);
    mat4_mul(out, a_i, t_next);
    memcpy(out, t_next, sizeof(t_next));
  }
}

static Vec3 fk_panda_position(const double q[7]) {
  double t[4][4];
  fk_panda(q, t);
  Vec3 p = {t[0][3], t[1][3], t[2][3]};
  return p;
}

// 3-DOF Jacobian (position only, using joints 1-4)
static void jacobian_numeric(const double q[7], double j[3][4]) {
  const double eps = 1e-4;
  Vec3 p0 = fk_panda_position(q);
  for (int i = 0; i < 4; ++i) {  // Only joints 1-4
    double q2[7];
    memcpy(q2, q, sizeof(q2));
    q2[i] += eps;
    Vec3 p1 = fk_panda_position(q2);
    j[0][i] = (p1.x - p0.x) / eps;
    j[1][i] = (p1.y - p0.y) / eps;
    j[2][i] = (p1.z - p0.z) / eps;
  }
}

static void write_capture_state(const char *path, int capturing, int frame_id) {
  FILE *file = fopen(path, "w");
  if (!file)
    return;
  fprintf(file, "capturing=%d\nframe=%d\n", capturing, frame_id);
  fclose(file);
}

int main(int argc, char **argv) {
  wb_robot_init();

  char device_name[32];

  /* --- Motors --- */
  for (int i = 0; i < 7; ++i) {
    sprintf(device_name, "panda_joint%d", i + 1);
    motors[i] = wb_robot_get_device(device_name);
    wb_motor_set_velocity(motors[i], 1.0);
    double min_pos = wb_motor_get_min_position(motors[i]);
    double max_pos = wb_motor_get_max_position(motors[i]);
    if (!isfinite(min_pos))
      min_pos = -3.14;
    if (!isfinite(max_pos))
      max_pos = 3.14;
    joint_min[i] = min_pos;
    joint_max[i] = max_pos;
  }

  motors[FINGER] = wb_robot_get_device("panda_finger::right");
  wb_motor_set_position(motors[FINGER], 0.02);  // open gripper

  /* --- Joint sensors --- */
  for (int i = 0; i < 7; ++i) {
    sprintf(device_name, "panda_joint%d_sensor", i + 1);
    sensors[i] = wb_robot_get_device(device_name);
    wb_position_sensor_enable(sensors[i], TIME_STEP);
  }

  /* --- Wrist camera --- */
  wrist_camera = wb_robot_get_device("wrist_camera");
  wb_camera_enable(wrist_camera, TIME_STEP);

  const char *world_path = wb_robot_get_world_path();
  char project_path[PATH_MAX] = {0};
  if (world_path != NULL) {
    char world_copy[PATH_MAX];
    snprintf(world_copy, sizeof(world_copy), "%s", world_path);
    char *world_dir = strrchr(world_copy, '/');
    if (world_dir != NULL) {
      *world_dir = '\0';
      char *project_dir = strrchr(world_copy, '/');
      if (project_dir != NULL) {
        *project_dir = '\0';
        snprintf(project_path, sizeof(project_path), "%s", world_copy);
      }
    }
  }
  if (project_path[0] == '\0') {
    if (getcwd(project_path, sizeof(project_path)) == NULL)
      snprintf(project_path, sizeof(project_path), ".");
  }

  char output_dir[512];
  char state_path[512];
  snprintf(output_dir, sizeof(output_dir), "%s/controllers/gripper_camera_logger/output", project_path);
  mkdir(output_dir, 0777);
  snprintf(state_path, sizeof(state_path), "%s/controllers/capture_state.txt", project_path);

  /* --- Main loop --- */
  wb_keyboard_enable(TIME_STEP);

  double targets[8] = {0.0};
  double gripper_target = 0.02;
  const double joint_step = 0.02;
  const double gripper_step = 0.002;
  const double gripper_min = 0.0;
  const double gripper_max = 0.04;

  // Initialize targets from current pose to prevent jumps.
  wb_robot_step(TIME_STEP);
  for (int i = 0; i < 7; ++i) {
    targets[i] = clamp(wb_position_sensor_get_value(sensors[i]), joint_min[i], joint_max[i]);
    wb_motor_set_position(motors[i], targets[i]);
  }

  // Store initial home position
  double home_position[7];
  for (int i = 0; i < 7; ++i) {
    home_position[i] = targets[i];
  }

  // Lock wrist joints 5 and 7, allow joint 6 (pitch) to move
  double locked_joint5 = targets[4];  // Joint 5 locked
  double locked_joint7 = targets[6];  // Joint 7 locked
  double wrist_pitch_target = targets[5];  // Joint 6 manual control

  printf("Cartesian velocity teleop controls:\n");
  printf("  X (left/right): Up/Down arrows\n");
  printf("  Y (forward/back): Left/Right arrows\n");
  printf("  Z (up/down): Z/X\n");
  printf("  Wrist Pitch: A (up) / S (down)\n");
  printf("  Gripper: O (open) / P (close)\n");
  printf("  Home: SPACE (return to start)\n");
  printf("  Capture: C (toggle)\n");

  int frame = 0;
  int capturing = 0;
  double last_toggle_time = -1.0;
  int returning_home = 0;  // Flag for home position return
  double home_start_time = -1.0;

  // Velocity-based control parameters
  const double ee_vel = 0.08;  // end-effector velocity (m/s)
  const double damp = 0.05;     // damping for Jacobian pseudoinverse (increased for stability)
  const double max_joint_vel = 0.5;  // max joint velocity (rad/s)
  const double wrist_pitch_step = 0.02;  // manual wrist pitch step
  
  while (wb_robot_step(TIME_STEP) != -1) {
    // Desired end-effector velocity (m/s)
    Vec3 desired_vel = {0.0, 0.0, 0.0};
    
    int key = wb_keyboard_get_key();
    while (key != -1) {
      switch (key) {
        case WB_KEYBOARD_LEFT:
          desired_vel.y = -ee_vel;
          break;
        case WB_KEYBOARD_RIGHT:
          desired_vel.y = ee_vel;
          break;
        case WB_KEYBOARD_UP:
          desired_vel.x = ee_vel;
          break;
        case WB_KEYBOARD_DOWN:
          desired_vel.x = -ee_vel;
          break;
        case 'Z':
        case 'z':
          desired_vel.z = ee_vel;
          break;
        case 'X':
        case 'x':
          desired_vel.z = -ee_vel;
          break;
        case 'A':
        case 'a':
          wrist_pitch_target += wrist_pitch_step;
          wrist_pitch_target = clamp(wrist_pitch_target, joint_min[5], joint_max[5]);
          break;
        case 'S':
        case 's':
          wrist_pitch_target -= wrist_pitch_step;
          wrist_pitch_target = clamp(wrist_pitch_target, joint_min[5], joint_max[5]);
          break;
        case 'O':
        case 'o':
          gripper_target = clamp(gripper_target + gripper_step, gripper_min, gripper_max);
          break;
        case 'P':
        case 'p':
          gripper_target = clamp(gripper_target - gripper_step, gripper_min, gripper_max);
          break;
        case 'C':
        case 'c': {
          double now = wb_robot_get_time();
          if (last_toggle_time < 0.0 || (now - last_toggle_time) > 0.3) {
            capturing = !capturing;
            printf("camera capture %s\n", capturing ? "started" : "stopped");
            write_capture_state(state_path, capturing, frame);
            last_toggle_time = now;
          }
          break;
        }
        case ' ': {
          // Return to home position
          printf("Returning to home position...\n");
          returning_home = 1;
          home_start_time = wb_robot_get_time();
          
          wb_motor_set_velocity(motors[0], 0.5);
          wb_motor_set_velocity(motors[1], 0.5);
          wb_motor_set_velocity(motors[2], 0.5);
          wb_motor_set_velocity(motors[3], 0.5);
          wb_motor_set_velocity(motors[4], 0.5);
          wb_motor_set_velocity(motors[5], 0.5);
          wb_motor_set_velocity(motors[6], 0.5);
          
          wb_motor_set_position(motors[0], home_position[0]);
          wb_motor_set_position(motors[1], home_position[1]);
          wb_motor_set_position(motors[2], home_position[2]);
          wb_motor_set_position(motors[3], home_position[3]);
          wb_motor_set_position(motors[4], home_position[4]);
          wb_motor_set_position(motors[5], home_position[5]);
          wb_motor_set_position(motors[6], home_position[6]);
          
          wrist_pitch_target = home_position[5];
          break;
        }
        default:
          break;
      }
      key = wb_keyboard_get_key();
    }

    // Read current joint positions
    double current_q[7];
    for (int i = 0; i < 7; ++i)
      current_q[i] = wb_position_sensor_get_value(sensors[i]);

    // Check if returning home is complete
    if (returning_home) {
      double elapsed = wb_robot_get_time() - home_start_time;
      if (elapsed > 3.0) {  // Allow 3 seconds for return
        returning_home = 0;
        printf("Home position reached.\n");
      } else {
        // Skip normal control while returning home
        if (capturing && frame % 10 == 0) {
          char filename[512];
          snprintf(filename, sizeof(filename), "%s/frame_%06d.png", output_dir, frame);
          wb_camera_save_image(wrist_camera, filename, 100);
          write_capture_state(state_path, capturing, frame);
        }
        frame++;
        continue;
      }
    }

    // Compute 3-DOF Jacobian for joints 1-4 only
    double j[3][4];
    jacobian_numeric(current_q, j);

    // Compute J * J^T (damped) - now 3x3 using 4 joints
    double jjt[3][3] = {{0.0}};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 4; ++k)
          jjt[r][c] += j[r][k] * j[c][k];
      }
      jjt[r][r] += damp * damp;  // damping
    }

    // Invert 3x3 matrix
    double det = jjt[0][0] * (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1]) -
                 jjt[0][1] * (jjt[1][0] * jjt[2][2] - jjt[1][2] * jjt[2][0]) +
                 jjt[0][2] * (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0]);
    
    double jjt_inv[3][3] = {{0.0}};
    if (fabs(det) > 1e-9) {
      double inv_det = 1.0 / det;
      jjt_inv[0][0] = (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1]) * inv_det;
      jjt_inv[0][1] = (jjt[0][2] * jjt[2][1] - jjt[0][1] * jjt[2][2]) * inv_det;
      jjt_inv[0][2] = (jjt[0][1] * jjt[1][2] - jjt[0][2] * jjt[1][1]) * inv_det;
      jjt_inv[1][0] = (jjt[1][2] * jjt[2][0] - jjt[1][0] * jjt[2][2]) * inv_det;
      jjt_inv[1][1] = (jjt[0][0] * jjt[2][2] - jjt[0][2] * jjt[2][0]) * inv_det;
      jjt_inv[1][2] = (jjt[0][2] * jjt[1][0] - jjt[0][0] * jjt[1][2]) * inv_det;
      jjt_inv[2][0] = (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0]) * inv_det;
      jjt_inv[2][1] = (jjt[0][1] * jjt[2][0] - jjt[0][0] * jjt[2][1]) * inv_det;
      jjt_inv[2][2] = (jjt[0][0] * jjt[1][1] - jjt[0][1] * jjt[1][0]) * inv_det;
    }

    // Compute J^T * (J*J^T)^-1 * desired_vel = joint velocities for joints 1-4
    double temp[3];
    temp[0] = jjt_inv[0][0] * desired_vel.x + jjt_inv[0][1] * desired_vel.y + jjt_inv[0][2] * desired_vel.z;
    temp[1] = jjt_inv[1][0] * desired_vel.x + jjt_inv[1][1] * desired_vel.y + jjt_inv[1][2] * desired_vel.z;
    temp[2] = jjt_inv[2][0] * desired_vel.x + jjt_inv[2][1] * desired_vel.y + jjt_inv[2][2] * desired_vel.z;

    double joint_vels[4];
    for (int i = 0; i < 4; ++i) {
      joint_vels[i] = j[0][i] * temp[0] + j[1][i] * temp[1] + j[2][i] * temp[2];
      // Clamp joint velocities
      if (joint_vels[i] > max_joint_vel)
        joint_vels[i] = max_joint_vel;
      if (joint_vels[i] < -max_joint_vel)
        joint_vels[i] = -max_joint_vel;
    }

    // Debug: Print joint velocities when moving
    static int debug_counter = 0;
    if ((desired_vel.x != 0.0 || desired_vel.y != 0.0 || desired_vel.z != 0.0) && debug_counter++ % 30 == 0) {
      printf("Joint velocities: J1=%.3f J2=%.3f J3=%.3f J4=%.3f\n",
             joint_vels[0], joint_vels[1], joint_vels[2], joint_vels[3]);
    }

    // Set motor velocities and compute target positions
    double dt = TIME_STEP / 1000.0;  // convert ms to seconds
    for (int i = 0; i < 7; ++i) {
      wb_motor_set_velocity(motors[i], max_joint_vel);
      double new_pos;
      
      if (i < 4) {
        // Joints 1-4: use IK velocities
        new_pos = current_q[i] + joint_vels[i] * dt;
        new_pos = clamp(new_pos, joint_min[i], joint_max[i]);
      } else if (i == 4) {
        // Joint 5: locked
        new_pos = locked_joint5;
      } else if (i == 5) {
        // Joint 6: manual control via A/S keys
        new_pos = wrist_pitch_target;
      } else {
        // Joint 7: locked
        new_pos = locked_joint7;
      }
      wb_motor_set_position(motors[i], new_pos);
    }
    wb_motor_set_position(motors[FINGER], gripper_target);

    if (capturing && frame % 10 == 0) {
      char filename[512];
      snprintf(filename, sizeof(filename), "%s/frame_%06d.png", output_dir, frame);
      wb_camera_save_image(wrist_camera, filename, 100);
      write_capture_state(state_path, capturing, frame);
    }
    frame++;
  }

  wb_robot_cleanup();
  return 0;
}
