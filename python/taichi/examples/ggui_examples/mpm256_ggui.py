#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from copy import deepcopy
import os
import gc
import numpy as np
import pdb
import pickle
import taichi as ti
import time
import argparse


# In[ ]:


def arg_parse():
    def str2bool(v):
        """used for argparse, 'type=str2bool', so that can pass in string True or False."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('true'):
            return True
        elif v.lower() in ('false'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Taichi argparse.')
    # Experiment management:
    parser.add_argument('--gravity_amp', type=float,
                        help='Gravity amplitude')
    parser.add_argument('--max_n_part_fluid', type=int,
                        help='Maximum number of fluid particles')
    parser.add_argument('--n_part_particle', type=int,
                        help='Number of particles')
    parser.add_argument('--n_grid', type=int,
                        help='Grid size. E.g. --n_grid=256 means the 2D space is 256x256.')
    parser.add_argument('--grid_width', type=float,
                        help='Width. By default is 1.')
    parser.add_argument('--n_steps', type=int,
                        help='Number of simulation steps')
    parser.add_argument('--is_particle', type=str2bool, nargs='?', const=True, default=True,
                        help="If True, will include particle")
    parser.add_argument('--is_gui', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will use GUI.')
    parser.add_argument('--n_simu', type=int,
                        help='Number of trajectories')
    parser.add_argument('--height', type=float,
                        help='Maximum height of the fluid.')
    parser.add_argument('--is_save', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will use GUI.')
    parser.add_argument('--gpuid', type=str,
                        help='GPU ID.')
    parser.add_argument('--traj_path', type=str,
                        help='trajectory path')
    parser.add_argument('--seed', type=int,
                        help='random seed.')
    parser.add_argument('--record_path', type=str,
                        help='Record path')

    parser.set_defaults(
        gravity_amp=2,
        max_n_part_fluid=30000,
        n_part_particle=1000,
        n_grid=256,
        grid_width=1.,
        n_steps=200,
        is_particle=True,
        is_gui=False,
        n_simu=500,
        height=0.25,
        is_save=True,
        gpuid="0",
        traj_path="/dfs/project/plasma/taichi_new_ggui/",
        seed=1,
        record_path="None",
    )
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args


def set_seed(seed):
    """Set up seed."""
    import numpy as np
    import torch
    import random
    if seed == -1:
        seed = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)


def pdump(file, filename):
    """Dump a file via pickle."""
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def pload(filename):
    """Load a filename saved as pickle."""
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file


def clear_dir(dirname):
    import os
    import glob
    files = glob.glob(dirname)
    for f in files:
        os.remove(f)


def record_data(data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False, recent_record=-1):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)
            if recent_record != -1:
                # Only keep the most recent records
                data_record_dict[key] = data_record_dict[key][-recent_record:]


args = arg_parse()
arch = ti.cuda if args.gpuid != "False" else ti.vulkan if ti._lib.core.with_vulkan() else ti.cpu
ti.init(arch=arch)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    is_jupyter = True
except:
    is_jupyter = False

gravity_amp = args.gravity_amp
max_n_part_fluid = args.max_n_part_fluid
n_part_particle = args.n_part_particle
n_grid = args.n_grid
grid_width = args.grid_width
n_grid_width = int(args.n_grid * args.grid_width)
n_steps = args.n_steps
is_particle = args.is_particle
is_gui = args.is_gui
n_simu = args.n_simu
height = args.height


# In[ ]:


@ti.kernel
def substep1(is_save_all: ti.i32):
    for i, j in grid_m:
        # gri_v & grid_m: [n_grid_width, n_grid]
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        if is_save_all == 1:
            grid_C[i, j] = [[0,0], [0,0]]
            grid_F[i, j] = [[0,0], [0,0]]
            grid_Jp[i, j] = 0
            grid_stress[i, j] = [[0,0], [0,0]]
            grid_affine[i, j] = [0,0]
    for p in x:  # x: [n_particles] Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)  # index of the grid
        fx = x[p] * inv_dx - base.cast(float)   # location in the grid
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))  #   Jp: [n_particles], plastic deformation
        if material[p] == 2:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 1:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 1:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        test_item[p] = stress
        
        # Loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
            if is_save_all == 1:
                grid_C[base + offset] += weight * C[p]
                grid_F[base + offset] += weight * F[p]
                grid_Jp[base + offset] += weight * Jp[p]
                grid_stress[base + offset] += weight * stress
                grid_affine[base + offset] += weight * (affine @ dpos)

@ti.kernel
def substep2():
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (
                0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            test_grid[i,j] = grid_v[i,j]
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid_width - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

@ti.kernel
def substep3():
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        # loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            outer = g_v.outer_product(dpos)
            new_C += 4 * inv_dx * weight * outer
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
        


@ti.kernel
def reset():
    for i in range(n_particles):
        epsilon = 1e-3
        if i < group_size:
            x[i] = [
                ti.random() * (grid_width - epsilon * 2) + epsilon,
                ti.random() * (0.2 - epsilon * 2) + epsilon
            ]
        else:
            x[i] = [
                (ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size)) * grid_width,
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
            ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def reset_other_fields(n_particles: int):
    for i in range(n_particles):
        material[i] = 0  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def render():
    for i in range(n_particles):
        if i < n_part_particle:
            particle[i] = x[i]
            v_particle[i] = v[i]
        else:
            fluid[i - n_part_particle] = x[i]
            v_fluid[i - n_part_particle] = v[i]


def sample_shape():
    y1 = np.random.rand()*0.05 + 0.01
    y2 = np.random.rand()*0.05 + height - 0.05
    if np.random.rand() > 0.5:
        y1, y2 = y2, y1
    shape = [
        ((0, y1), (grid_width, y2), 0),
    ]
    return shape


def get_fluid(fluid_shape, n_part, height, epsilon=1e-3):
    x_np = np.stack([
        np.random.rand(n_part) * (grid_width - epsilon * 2) + epsilon,
        np.random.rand(n_part) * height + epsilon,
    ], -1)
    lx = x_np[:, 0]
    ly = x_np[:, 1]
    mask = np.ones(len(x_np)).astype(bool)
    for shape_ele in fluid_shape:
        (x1, y1), (x2, y2), direction = shape_ele
        out = (ly - y1) - (y2 - y1)/(x2 - x1)*(lx - x1)
        if direction == 1:
            mask_ele = out >= 0
        elif direction == 0:
            mask_ele = out <= 0
        else:
            raise
        mask = mask & mask_ele
    x_fluid = x_np[mask]
    return x_fluid


def sample_rect_shape():
    x1 = (np.random.rand() * 0.2 + 0.1) * grid_width
    x2 = (-np.random.rand() * 0.2 + 0.9) * grid_width
    y1 = np.random.rand() * 0.2 + 0.4
    y2 = np.random.rand() * 0.2 + 0.8
    rect_shape = (x1, y1), (x2, y2)
    return rect_shape


def get_particles(rect_shape, n_part):
    """
    lowerleft: (x1, y1)
    upperright: (x2, y2)
    """
    (x1, y1), (x2, y2) = rect_shape
    lx_np = np.random.rand(n_part, 1) * (x2 - x1) + x1
    ly_np = np.random.rand(n_part, 1) * (y2 - y1) + y1
    x_np = np.concatenate([lx_np, ly_np], -1)
    return x_np

def plot_part(x):
    import matplotlib.pylab as plt
    if not isinstance(x, np.ndarray):
        x = x.to_numpy()
    plt.plot(x[:,0], x[:,1], ".")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


def remove_too_near(x_np, threshold, mode="simu"):
    def subfun(x_np):
        dist = np.sqrt(((x_np[None] - x_np[:,None]) ** 2).sum(-1))
        length = len(dist)
        idx = np.arange(length)
        dist[idx,idx] = 1
        dist_min = dist.min(-1)
        idx_sort = np.argsort(dist_min)
        return dist_min, idx_sort
    length = len(x_np)
    if mode == "seq":
        for i in range(length):
            if i % 100 == 0:
                print(i)
            dist_min, idx_sort = subfun(x_np)
            if dist_min[idx_sort[0]] < threshold:
                mask = np.ones(len(x_np)).astype(bool)
                mask[idx_sort[0]] = False
                x_np = x_np[mask]
            else:
                break
    elif mode == "simu":
        dist_min, idx_sort = subfun(x_np)
        mask = dist_min >= threshold
        x_np = x_np[mask]
    else:
        raise
    return x_np


# In[ ]:


def get_trajectory(x, v, C, F, material, Jp, fluid, v_fluid, particle, v_particle, grid_m, grid_v, grid_C, grid_F, grid_Jp, grid_stress, grid_affine, is_gui=True, record_path="None"):
    data_record = {}
    data_record["x_fluid"] = -np.ones((n_steps, fluid.shape[0], 2))
    data_record["v_fluid"] = -np.ones((n_steps, fluid.shape[0], 2))
    data_record["x_particle"] = -np.ones((n_steps, particle.shape[0], 2))
    data_record["v_particle"] = -np.ones((n_steps, particle.shape[0], 2))
    data_record["grid_m"] = -np.ones((n_steps, n_grid_width, n_grid))
    data_record["grid_v"] = -np.ones((n_steps, n_grid_width, n_grid, 2))
    data_record["grid_C"] = -np.ones((n_steps, n_grid_width, n_grid, 2, 2))
    data_record["grid_F"] = -np.ones((n_steps, n_grid_width, n_grid, 2, 2))
    data_record["grid_Jp"] = -np.ones((n_steps, n_grid_width, n_grid))
    data_record["grid_stress"] = -np.ones((n_steps, n_grid_width, n_grid, 2, 2))
    data_record["grid_affine"] = -np.ones((n_steps, n_grid_width, n_grid, 2))
    data_record["n_part_fluid"] = fluid.shape[0]
    data_record["n_part_particle"] = particle.shape[0]
    n_particles = particle.shape[0] + fluid.shape[0]
    data_record["x"] = -np.ones((n_steps, n_particles, 2))
    data_record["v"] = -np.ones((n_steps, n_particles, 2))
    data_record["C"] = -np.ones((n_steps, n_particles, 2, 2))
    data_record["F"] = -np.ones((n_steps, n_particles, 2, 2))
    data_record["Jp"] = -np.ones((n_steps, n_particles))
    if is_gui:
        res = (512, 512)
        window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
        canvas = window.get_canvas()
        radius = 0.003

    gravity[None] = [0, -1]
    k = 0
    while True:
        if is_gui:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    x, v, C, F, material, Jp, fluid, particle, n_particles, n_part_fluid = reset_all(particle.shape[0])
                elif window.event.key in [ti.ui.ESCAPE]:
                    break
        gravity[None][1] = -gravity_amp
        total_substeps = int(2e-3 // dt)
        for s in range(total_substeps):
            substep1(is_save_all=1 if s==total_substeps-1 else 0)
            if record_path != "None" and not os.path.isfile(record_path + f"/mpm256_ori/{s}_substep1.p"):
                record_dict = {}
                record_data(record_dict, 
                            list(deepcopy((x.to_numpy(), v.to_numpy(), C.to_numpy(), F.to_numpy(), Jp.to_numpy(), grid_m.to_numpy(), grid_v.to_numpy(), grid_C.to_numpy(), grid_F.to_numpy(), grid_Jp.to_numpy(), grid_stress.to_numpy(), grid_affine.to_numpy(), test_item.to_numpy()))),
                            ["x", "v", "C", "F", "Jp", "grid_m", "grid_v", "grid_C", "grid_F", "grid_Jp", "grid_stress", "grid_affine", "test_item"], nolist=True)
                make_dir(record_path + "/mpm256_ori/test")
                pdump(record_dict, record_path + f"/mpm256_ori/{s}_substep1.p")
            substep2()
            if record_path != "None" and not os.path.isfile(record_path + f"/mpm256_ori/{s}_substep2.p"):
                record_dict = {}
                record_data(record_dict, 
                            list(deepcopy((x.to_numpy(), v.to_numpy(), C.to_numpy(), F.to_numpy(), Jp.to_numpy(), grid_m.to_numpy(), grid_v.to_numpy(), grid_C.to_numpy(), grid_F.to_numpy(), grid_Jp.to_numpy(), grid_stress.to_numpy(), grid_stress.to_numpy(), test_item.to_numpy()))),
                            ["x", "v", "C", "F", "Jp", "grid_m", "grid_v", "grid_C", "grid_F", "grid_Jp", "grid_stress", "grid_affine", "test_item"], nolist=True)
                pdump(record_dict, record_path + f"/mpm256_ori/{s}_substep2.p")
            substep3()
            if record_path != "None" and not os.path.isfile(record_path + f"/mpm256_ori/{s}_substep3.p"):
                record_dict = {}
                record_data(record_dict, 
                            list(deepcopy((x.to_numpy(), v.to_numpy(), C.to_numpy(), F.to_numpy(), Jp.to_numpy(), grid_m.to_numpy(), grid_v.to_numpy(), grid_C.to_numpy(), grid_F.to_numpy(), grid_Jp.to_numpy(), grid_stress.to_numpy(), grid_stress.to_numpy(), test_item.to_numpy()))),
                            ["x", "v", "C", "F", "Jp", "grid_m", "grid_v", "grid_C", "grid_F", "grid_Jp", "grid_stress", "grid_affine", "test_item"], nolist=True)
                pdump(record_dict, record_path + f"/mpm256_ori/{s}_substep3.p")
        render()
        if is_gui:
            canvas.set_background_color((0.067, 0.184, 0.255))
            canvas.circles(fluid, radius=radius, color=(0, 0.5, 0.5))
            # # canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
            canvas.circles(particle, radius=radius, color=(0, 0.5, 0.5))
            window.show()
        data_record["x_fluid"][k] = fluid.to_numpy()
        data_record["v_fluid"][k] = v_fluid.to_numpy()
        data_record["x_particle"][k] = particle.to_numpy()
        data_record["v_particle"][k] = v_particle.to_numpy()
        data_record["grid_m"][k] = grid_m.to_numpy()
        data_record["grid_v"][k] = grid_v.to_numpy()
        data_record["grid_C"][k] = grid_C.to_numpy()
        data_record["grid_F"][k] = grid_F.to_numpy()
        data_record["grid_Jp"][k] = grid_Jp.to_numpy()
        data_record["grid_stress"][k] = grid_stress.to_numpy()
        data_record["grid_affine"][k] = grid_affine.to_numpy()
        data_record["x"][k] = x.to_numpy()
        data_record["v"][k] = v.to_numpy()
        data_record["C"][k] = C.to_numpy()
        data_record["F"][k] = F.to_numpy()
        data_record["Jp"][k] = Jp.to_numpy()
        k += 1
        if k >= n_steps:
            break
    return data_record


# In[ ]:


def reset_all(n_part_particle):
    fluid_shape = sample_shape()
    x_fluid = get_fluid(fluid_shape, n_part=max_n_part_fluid, height=height).astype(np.float32)
    n_part_fluid = len(x_fluid)
    rect_shape = sample_rect_shape()
    x_particle = get_particles(rect_shape, n_part=n_part_particle).astype(np.float32)
    print("fluid:", n_part_fluid)
    print("part:", n_part_particle)

    # # Remove too near:
    if threshold > 0:
        x_particle = remove_too_near(x_particle, threshold=threshold)
        x_fluid = remove_too_near(x_fluid, threshold=threshold)
        n_part_fluid = len(x_fluid)
        print("fluid, after:", n_part_fluid)
        n_part_particle = len(x_particle)
        print("part, after:", n_part_particle)

    if is_particle:
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
        n_particles = n_part_fluid + n_part_particle
    else:
        n_part_particle = 1
        x_particle = np.array([[0.001, 0.001]]).astype(np.float32)
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
        n_particles = n_part_fluid + n_part_particle


    x = ti.Vector.field(2, dtype=float, shape=n_particles)
    x.from_numpy(x_combine)
    fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
    particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)
    fluid.from_numpy(x_fluid)
    particle.from_numpy(x_particle)
    v_fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
    v_particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)

    # Initialize up other fields:
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

    reset_other_fields(n_particles)

    return x, v, C, F, material, Jp, fluid, v_fluid, particle, v_particle, rect_shape, fluid_shape


# In[ ]:


threshold = 0
if args.record_path != "None":
    clear_dir(args.record_path + "/mpm256_ori/*")

for ll in range(n_simu):
    set_seed(args.seed + ll)
    ti.init(arch=arch)
    
    quality = 1  # Use a larger value for higher-res simulations
    n_particles, n_grid = 9000 * quality**2, n_grid * quality
    dx, inv_dx = 1 / n_grid, float(n_grid)
    dt = 5e-5 / quality
    p_vol, p_rho = (dx * 0.5)**2, 1
    p_mass = p_vol * p_rho
    E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters

    x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
    grid_m = ti.field(dtype=float, shape=(n_grid_width, n_grid))  # grid node mass
    grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid_width, n_grid))  # grid node momentum/velocity
    grid_C = ti.Matrix.field(2, 2, dtype=float, shape=(n_grid_width, n_grid))
    grid_F = ti.Matrix.field(2, 2, dtype=float, shape=(n_grid_width, n_grid))
    grid_Jp = ti.field(dtype=float, shape=(n_grid_width, n_grid))
    grid_stress = ti.Matrix.field(2, 2, dtype=float, shape=(n_grid_width, n_grid))
    grid_affine = ti.Vector.field(2, dtype=float, shape=(n_grid_width, n_grid))
    gravity = ti.Vector.field(2, dtype=float, shape=())
    attractor_strength = ti.field(dtype=float, shape=())
    attractor_pos = ti.Vector.field(2, dtype=float, shape=())

    group_size = n_particles // 2
    water = ti.Vector.field(2, dtype=float, shape=group_size)  # position
    jelly = ti.Vector.field(2, dtype=float, shape=group_size)  # position
    snow = ti.Vector.field(2, dtype=float, shape=group_size)  # position
    mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))


    print(f"Simu: {ll}")

    fluid_shape = sample_shape()
    x_fluid = get_fluid(fluid_shape, n_part=max_n_part_fluid, height=height).astype(np.float32)
    n_part_fluid = len(x_fluid)
    rect_shape = sample_rect_shape()
    x_particle = get_particles(rect_shape, n_part=n_part_particle).astype(np.float32)
    print("fluid:", n_part_fluid)
    print("part:", n_part_particle)

    # # Remove too near:
    if threshold > 0:
        x_particle = remove_too_near(x_particle, threshold=threshold)
        x_fluid = remove_too_near(x_fluid, threshold=threshold)
        n_part_fluid = len(x_fluid)
        print("fluid, after:", n_part_fluid)
        n_part_particle = len(x_particle)
        print("part, after:", n_part_particle)

    if is_particle:
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
        n_particles = n_part_fluid + n_part_particle
    else:
        n_part_particle = 1
        x_particle = np.array([[0.001, 0.001]]).astype(np.float32)
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
        n_particles = n_part_fluid + n_part_particle

    data_dirname = f"taichi_hybrid_simu_{n_simu}_step_{n_steps}_h_{height}_fluid_{max_n_part_fluid}_part_{n_part_particle}_g_{gravity_amp}_thresh_{threshold}"

    x = ti.Vector.field(2, dtype=float, shape=n_particles)
    x.from_numpy(x_combine)
    fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
    particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)
    fluid.from_numpy(x_fluid)
    particle.from_numpy(x_particle)
    v_fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
    v_particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)

    # Initialize up other fields:
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
    test_item = ti.Matrix.field(2,2, dtype=float, shape=n_particles)  # plastic deformation
    test_grid = ti.Vector.field(2, dtype=float, shape=(n_grid_width, n_grid))

    # Reset other fields:
    reset_other_fields(n_particles)

    # x, v, C, F, material, Jp, fluid, v_fluid, particle, v_particle, rect_shape, fluid_shape = reset_all(n_part_particle)

    # Get trajectory:
    data_record = get_trajectory(
        x=x,
        v=v,
        C=C,
        F=F,
        material=material,
        Jp=Jp,
        fluid=fluid,
        v_fluid=v_fluid,
        particle=particle,
        v_particle=v_particle,
        grid_m=grid_m,
        grid_v=grid_v,
        grid_C=grid_C,
        grid_F=grid_F,
        grid_Jp=grid_Jp,
        grid_stress=grid_stress,
        grid_affine=grid_affine,
        is_gui=is_gui,
        record_path=args.record_path,
    )
    data_record["rect_shape"] = rect_shape
    data_record["fluid_shape"] = fluid_shape

    if args.is_save:
        data_dirname = os.path.join(args.traj_path, f"taichi_hybrid_simu_{n_simu}_step_{n_steps}_h_{height}_fluid_{max_n_part_fluid}_part_{particle.shape[0]}_g_{gravity_amp}_thresh_{threshold}_gridwidth_{grid_width}")
        data_filename = data_dirname + "/sim_{:06d}.p".format(ll)
        make_dir(data_filename)
        pickle.dump(data_record, open(data_filename, "wb"))
    del x, v, fluid, particle, v_fluid, v_particle, C, F, material, Jp, data_record
    gc.collect()
    print()
    ti.reset()