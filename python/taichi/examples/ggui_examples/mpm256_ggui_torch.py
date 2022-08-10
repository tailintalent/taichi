#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import numpy as np
import pdb
import taichi as ti
import pickle
from numbers import Number
import os, sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)


# ## Functions:

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
    parser.add_argument('--n_steps', type=int,
                        help='Number of simulation steps')
    parser.add_argument('--is_particle', type=str2bool, nargs='?', const=True, default=True,
                        help="If True, will include particle")
    parser.add_argument('--is_gui', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will use GUI.')
    parser.add_argument('--n_simu', type=int,
                        help='Number of trajectories')
    parser.add_argument('--epsilon', type=float,
                        help='Threshold for grid_m')
    parser.add_argument('--height', type=float,
                        help='Maximum height of the fluid.')
    parser.add_argument('--gpuid', type=str,
                        help='GPU ID.')
    parser.add_argument('--is_save', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will use GUI.')
    parser.add_argument('--seed', type=int,
                        help='random seed.')

    parser.set_defaults(
        gravity_amp=2,
        max_n_part_fluid=30000,
        n_part_particle=1000,
        n_grid=128,
        n_steps=200,
        is_particle=True,
        is_gui=False,
        n_simu=500,
        epsilon=1e-7,
        height=0.25,
        is_save=True,
        gpuid="0",
        seed=1,
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


args = arg_parse()
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    is_jupyter = True
    # args.n_simu = 50
    args.is_save = False
    args.gpuid = "False"
    args.epsilon=1e-6
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..', '..', '..', '..'))
    from plasma.pytorch_net.util import plot_matrices, pload, pdump
except:
    is_jupyter = False

gravity_amp = args.gravity_amp
max_n_part_fluid = args.max_n_part_fluid
n_part_particle = args.n_part_particle
n_grid = args.n_grid
n_steps = args.n_steps
is_particle = args.is_particle
is_gui = args.is_gui
n_simu = args.n_simu
height = args.height
if args.gpuid == "False":
    device = "cpu"
else:
    device = f"cuda:{args.gpuid}"


# In[ ]:


quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, n_grid * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

# x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
# v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
# C = ti.Matrix.field(2, 2, dtype=float,
#                     shape=n_particles)  # affine velocity field
# F = ti.Matrix.field(2, 2, dtype=float,
#                     shape=n_particles)  # deformation gradient
# material = ti.field(dtype=int, shape=n_particles)  # material id
# Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

grid_v = torch.zeros(n_grid, n_grid, 2, device=device, dtype=torch.float64)  # grid node momentum/velocity
grid_m = torch.zeros(n_grid, n_grid, device=device, dtype=torch.float64)  # grid node mass
gravity = torch.zeros(2, device=device, dtype=torch.float64)
attractor_strength = torch.tensor(0., device=device, dtype=torch.float64)
attractor_pos = torch.zeros(2, device=device, dtype=torch.float64)

# group_size = n_particles // 2
# water = ti.Vector.field(2, dtype=float, shape=group_size)  # position
# jelly = ti.Vector.field(2, dtype=float, shape=group_size)  # position
# snow = ti.Vector.field(2, dtype=float, shape=group_size)  # position
# mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))


# In[ ]:


@ti.kernel
def substep():
    # for i, j in grid_m:
    #     # gri_v & grid_m: [n_grd, n_grid]
    #     grid_v[i, j] = [0, 0]
    #     grid_m[i, j] = 0
    grid_v.fill_(0)
    grid_m.fill_(0)

    base = (x * inv_dx - 0.5).long()  # [n_particles] index of the grid
    fx = x * inv_dx - base   # [n_particles] location in the grid
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]  # [n_particles] 
    F = (torch.eye(2)[None].expand(n_particles,2,2) + C * dt) @ F
    
    
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
        # Loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (
                0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
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
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    for i in range(n_particles):
        if i < group_size:
            x[i] = [
                ti.random() * 1,
                ti.random() * 0.2
            ]
        else:
            x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
            ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


# @ti.kernel
def reset_other_fields(n_particles: int):
    material = torch.zeros(n_particles, device=device, dtype=torch.float64)
    v = torch.zeros(n_particles, 2, device=device, dtype=torch.float64)
    F = torch.tensor([[1, 0], [0, 1]], device=device, dtype=torch.float64)[None].expand(n_particles, 2, 2)
    Jp = torch.ones(n_particles, device=device, dtype=torch.float64)
    C = torch.zeros(n_particles, 2, 2, device=device, dtype=torch.float64)
    return v, C, F, material, Jp


# @ti.kernel
# def render():
#     for i in range(n_particles):
#         if i < n_part_particle:
#             particle_ti[i] = x[i]
#             v_particle_ti[i] = v[i]
#         else:
#             fluid_ti[i - n_part_particle] = x[i]
#             v_fluid_ti[i - n_part_particle] = v[i]

def render(x, v, n_part_particle):
    particle = x[:n_part_particle]
    v_particle = v[:n_part_particle]
    fluid = x[n_part_particle:]
    v_fluid = v[n_part_particle:]
    return particle, v_particle, fluid, v_fluid


def sample_shape():
    y1 = np.random.rand()*0.05 + 0.01
    y2 = np.random.rand()*0.05 + height - 0.05
    if np.random.rand() > 0.5:
        y1, y2 = y2, y1
    shape = [
        ((0, y1), (1, y2), 0),
    ]
    return shape


def get_fluid(fluid_shape, n_part, height, epsilon=1e-3):
    x_np = np.stack([
        np.random.rand(n_part) * (1 - epsilon * 2) + epsilon,
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
    x1 =  np.random.rand() * 0.2 + 0.1
    x2 = -np.random.rand() * 0.2 + 0.9
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


def expand_grid(grid, length):
    if len(grid.shape) == 3:
        n_features = grid.shape[-1]
        grid_expand = torch.cat([torch.cat([grid, torch.zeros(n_grid,length,n_features, device=device, dtype=torch.float64)], 1), torch.zeros(length,n_grid+length,n_features, device=device, dtype=torch.float64)], 0)
    elif len(grid.shape) == 2:
        grid_expand = torch.cat([torch.cat([grid, torch.zeros(n_grid,length, device=device, dtype=torch.float64)], 1), torch.zeros(length,n_grid+length, device=device, dtype=torch.float64)], 0)
    else:
        raise
    return grid_expand


def shrink_grid(grid, length):
    return grid[:-length, :-length]

def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if array is None:
            array_list.append(array)
            continue
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or            isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        if not ("keep_list" in kwargs and kwargs["keep_list"]):
            array_list = array_list[0]
    return array_list


# ## Functions torch:

# In[ ]:


def substep_torch1(x, v, C, F, Jp, gravity, epsilon=1e-6):
    grid_v = torch.zeros(n_grid, n_grid, 2, device=device, dtype=torch.float64)  # grid node momentum/velocity
    grid_m = torch.zeros(n_grid, n_grid, device=device, dtype=torch.float64)  # grid node mass

    base = (x * inv_dx - 0.5).long()  # [n_particles] index of the grid
    fx = x * inv_dx - base   # [n_particles] location in the grid
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]  # [n_particles] 
    F = (torch.eye(2, device=device, dtype=torch.float64)[None].expand(n_particles,2,2) + C * dt) @ F
    # Hardening coefficient: snow gets harder when compressed
    h = torch.maximum(torch.tensor(0.1, device=device), torch.minimum(torch.tensor(5, device=device), torch.exp(10 * (1.0 - Jp))))  #   Jp: [n_particles], plastic deformation
    mask_material_0 = (material==0)
    mask_material_1 = (material==1)
    mask_material_2 = (material==2)
    h[mask_material_2] = 0.3
    mu, la = mu_0 * h, lambda_0 * h
    length = len(F)
    mu[mask_material_0] = 0.  # liquid

    U, sig, V = torch.svd(F)
    J = 1.0
    for ii in range(2):
        new_sig = sig[:, ii] # sig: [n_particles, 2]
        new_sig[mask_material_1] = torch.minimum(torch.maximum(sig[mask_material_1, ii], torch.tensor(1 - 2.5e-2)), torch.tensor(1 + 4.5e-3))  # Plasticity
        Jp = Jp * sig[:, ii] / new_sig
        sig[:, ii] = new_sig
        J = J * new_sig
    F[mask_material_0] = torch.eye(2, device=device, dtype=torch.float64)[None].expand(len(mask_material_0),2,2) * J[:,None,None].sqrt()
    sig_matrix = torch.eye(2, device=device, dtype=torch.float64)[None].expand(length,2,2) * sig[...,None]
    F[mask_material_1] = U[mask_material_1] @ sig_matrix[mask_material_1] @ V[mask_material_1].transpose(1,2)

    stress = 2 * mu[:,None,None] * (F - U @ V.transpose(1,2)) @ F.transpose(1,2
            ) + torch.eye(2, device=device, dtype=torch.float64)[None].expand(length,2,2) * (la * J * (J - 1))[:,None,None]
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C

    # Deposition onto grid:
    # Loop over 3x3 grid node neighborhood
    grid_v_expand = expand_grid(grid_v, length=2)
    grid_m_expand = expand_grid(grid_m, length=2)
    for i in range(3):
        for j in range(3):
            offset = torch.tensor([i, j], device=device, dtype=torch.float64)
            dpos = (offset - fx) * dx
            weight = w[i][:,0] * w[j][:,1]
            base_core = base + offset.long()
            try:
                grid_v_expand[base_core[:,0],base_core[:,1]] += weight[:,None] * (p_mass * v + (affine @ dpos[:,:,None]).squeeze(-1))
            except:
                pdb.set_trace()
            grid_m_expand[base_core[:,0],base_core[:,1]] += weight * p_mass

    grid_v = shrink_grid(grid_v_expand, length=2)
    grid_m = shrink_grid(grid_m_expand, length=2)
    return x, v, C, F, Jp, grid_m, grid_v

    
def substep_torch2(x, v, C, F, Jp, grid_m, grid_v, gravity, epsilon=1e-6):
    # Boundary condition:
    rows, cols = torch.where(grid_m>epsilon)  # No need for epsilon here
    grid_v[rows,cols] = (1 / grid_m[rows,cols][...,None]) * grid_v[rows,cols]  # Momentum to velocity
    grid_v[rows,cols] += dt * gravity[None] * 30  # gravity
    mask_mg0 = (grid_m>0)
    mask_i0 = (grid_v[:,:,0] < 0) & torch.cat([torch.ones(3, n_grid, device=device, dtype=torch.float64), torch.zeros(n_grid-3, n_grid, device=device, dtype=torch.float64)], 0).bool()
    mask_i1 = (grid_v[:,:,0] > 0) & torch.cat([torch.zeros(n_grid-3, n_grid, device=device, dtype=torch.float64), torch.ones(3, n_grid, device=device, dtype=torch.float64)], 0).bool()
    mask_i  = mask_mg0 & (mask_i0 | mask_i1)
    grid_v[...,0].masked_fill_(mask_i, 0)
    mask_j0 = (grid_v[:,:,1] < 0) & torch.cat([torch.ones(n_grid, 3, device=device, dtype=torch.float64), torch.zeros(n_grid, n_grid-3, device=device, dtype=torch.float64)], 1).bool()
    mask_j1 = (grid_v[:,:,1] > 0) & torch.cat([torch.zeros(n_grid, n_grid-3, device=device, dtype=torch.float64), torch.ones(n_grid, 3, device=device, dtype=torch.float64)], 1).bool()
    mask_j  = mask_mg0 & (mask_j0 | mask_j1)
    grid_v[...,1].masked_fill_(mask_j, 0)
    return x, v, C, F, Jp, grid_m, grid_v


def substep_torch3(x, v, C, F, Jp, grid_m, grid_v, gravity, epsilon=1e-6):
    # Grid to particle (G2P)
    length = len(F)
    base = (x * inv_dx - 0.5).long()
    fx = x * inv_dx - base.double()
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
    new_v = torch.zeros(length, 2, device=device, dtype=torch.float64)
    new_C = torch.zeros(length, 2, 2, device=device, dtype=torch.float64)
    grid_v_expand = expand_grid(grid_v, length=2)
    for i in range(3):
        for j in range(3):
            dpos = torch.tensor([i, j], device=device, dtype=torch.float64) - fx
            base_core = base + torch.tensor([i, j], device=device)
            g_v = grid_v_expand[base_core[:,0],base_core[:,1]]
            weight = w[i][:,0] * w[j][:,1]
            new_v += weight[...,None] * g_v
            new_C += 4 * inv_dx * weight[:,None,None] * torch.einsum("bi,bj->bij", g_v, dpos)
    v, C = new_v, new_C
    x += dt * v
    return x, v, C, F, Jp, grid_m, grid_v


# In[ ]:


def get_trajectory(x, fluid, v_fluid, particle, v_particle, grid_m, grid_v, n_particles, epsilon=1e-8, is_gui=True):
    # Reset other fields:
    v, C, F, material, Jp = reset_other_fields(n_particles)
    n_part_fluid = len(fluid)
    n_part_particle = len(particle)

    # Initialize data_record:
    data_record = {}
    data_record["x_fluid"] = -np.ones((n_steps, fluid.shape[0], 2))
    data_record["v_fluid"] = -np.ones((n_steps, fluid.shape[0], 2))
    data_record["x_particle"] = -np.ones((n_steps, particle.shape[0], 2))
    data_record["v_particle"] = -np.ones((n_steps, particle.shape[0], 2))
    data_record["grid_m"] = -np.ones((n_steps, n_grid, n_grid))
    data_record["grid_v"] = -np.ones((n_steps, n_grid, n_grid, 2))
    data_record["n_part_fluid"] = fluid.shape[0]
    data_record["n_part_particle"] = particle.shape[0]
    if is_gui:
        res = (512, 512)
        window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
        canvas = window.get_canvas()
        radius = 0.003
        fluid_ti = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
        particle_ti = ti.Vector.field(2, dtype=float, shape=n_part_particle)

    gravity = torch.tensor([0, -1], dtype=torch.float64, device=device)
    k = 0
    while True:
        if is_gui:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    x, v, C, F, material, Jp, fluid, particle, n_particles, n_part_fluid = reset_all()
                elif window.event.key in [ti.ui.ESCAPE]:
                    break
        gravity[1] = -gravity_amp
        for s in range(int(2e-3 // dt)):
            x, v, C, F, Jp, grid_m, grid_v = substep_torch1(x, v, C, F, Jp, gravity, epsilon=epsilon)
            x, v, C, F, Jp, grid_m, grid_v = substep_torch2(x, v, C, F, Jp, grid_m, grid_v, gravity, epsilon=epsilon)
            x, v, C, F, Jp, grid_m, grid_v = substep_torch3(x, v, C, F, Jp, grid_m, grid_v, gravity, epsilon=epsilon)
        particle, v_particle, fluid, v_fluid = render(x, v, n_part_particle)
        if is_gui:
            fluid_ti.from_numpy(fluid)
            particle_ti.from_numpy(particle)
            canvas.set_background_color((0.067, 0.184, 0.255))
            canvas.circles(fluid_ti, radius=radius, color=(0, 0.5, 0.5))
            # # canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
            canvas.circles(particle_ti, radius=radius, color=(0, 0.5, 0.5))
            window.show()
        data_record["x_fluid"][k] = to_np_array(fluid)
        data_record["v_fluid"][k] = to_np_array(v_fluid)
        data_record["x_particle"][k] = to_np_array(particle)
        data_record["v_particle"][k] = to_np_array(v_particle)
        data_record["grid_m"][k] = to_np_array(grid_m)
        data_record["grid_v"][k] = to_np_array(grid_v)
        k += 1
        # if k % 10 == 0:
        #     print(k)
        if k >= n_steps:
            break
    return data_record


# ## Simulation:

# In[ ]:


threshold = 0
set_seed(args.seed)

for ll in range(n_simu):
    print(f"Simu: {ll}")

    fluid_shape = sample_shape()
    x_fluid = get_fluid(fluid_shape, n_part=max_n_part_fluid, height=height).astype(np.float64)
    n_part_fluid = len(x_fluid)
    rect_shape = sample_rect_shape()
    x_particle = get_particles(rect_shape, n_part=n_part_particle).astype(np.float64)
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
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float64)
        n_particles = n_part_fluid + n_part_particle
    else:
        n_part_particle = 1
        x_particle = np.array([[0.001, 0.001]]).astype(np.float64)
        x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float64)
        n_particles = n_part_fluid + n_part_particle

    data_dirname = f"taichi_hybrid_simu_{n_simu}_step_{n_steps}_h_{height}_fluid_{max_n_part_fluid}_part_{n_part_particle}_g_{gravity_amp}_thresh_{threshold}"

    x = torch.tensor(x_combine, device=device, dtype=torch.float64)
    fluid = torch.tensor(x_fluid, device=device, dtype=torch.float64)
    particle = torch.tensor(x_particle, device=device, dtype=torch.float64)
    v_fluid = torch.zeros(n_part_fluid, 2, device=device, dtype=torch.float64)
    v_particle = torch.zeros(n_part_particle, 2, device=device, dtype=torch.float64)

    # Initialize up other fields:
    v = torch.zeros(n_particles, 2, device=device, dtype=torch.float64)  # velocity
    C = torch.zeros(n_particles, 2, 2, device=device, dtype=torch.float64)  # affine velocity field
    F = torch.zeros(n_particles, 2, 2, device=device, dtype=torch.float64)  # deformation gradient
    material = torch.zeros(n_particles, device=device).long()  # material id
    Jp = torch.zeros(n_particles, device=device, dtype=torch.float64)  # plastic deformation

    # Get trajectory:
    data_record = get_trajectory(
        x, fluid, v_fluid, particle, v_particle, grid_m, grid_v, n_particles, epsilon=args.epsilon, is_gui=is_gui)
    data_record["rect_shape"] = rect_shape
    data_record["fluid_shape"] = fluid_shape

    if args.is_save:
        data_dirname = f"taichi_hybrid_simu_{n_simu}_step_{n_steps}_h_{height}_fluid_{max_n_part_fluid}_part_{particle.shape[0]}_g_{gravity_amp}_thresh_{threshold}"
        data_filename = data_dirname + "/sim_{:06d}.p".format(ll)
        make_dir(data_filename)
        pickle.dump(data_record, open(data_filename, "wb"))
    del x, v, fluid, particle, v_fluid, v_particle, C, F, material, Jp, data_record
    gc.collect()
    print()

