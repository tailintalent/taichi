#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import taichi as ti
import numpy as np
import time
import pdb

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
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
grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

group_size = n_particles // 2
water = ti.Vector.field(2, dtype=float, shape=group_size)  # position
jelly = ti.Vector.field(2, dtype=float, shape=group_size)  # position
snow = ti.Vector.field(2, dtype=float, shape=group_size)  # position
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))


# In[ ]:


@ti.kernel
def substep():
    for i, j in grid_m:
        # gri_v & grid_m: [n_grd, n_grid]
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
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
        else:
            fluid[i - n_part_particle] = x[i]


def sample_shape():
    y1 = np.random.rand()*0.15 + 0.05
    y2 = np.random.rand()*0.15 + 0.05
    shape = [
        ((0, y1), (1, y2), 0),
    ]
    return shape


def get_fluid(fluid_shape, n_part):
    height = 0.2
    x_np = np.stack([
        np.random.rand(n_part) * 1,
        np.random.rand(n_part) * height,
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
    x1 = np.random.rand() * 0.2 + 0.25
    x2 = np.random.rand() * 0.2 + 0.75
    y1 = np.random.rand() * 0.2 + 0.2
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


# In[ ]:


def reset_all():
    # Set up x:
    fluid_shape = sample_shape()
    x_fluid = get_fluid(fluid_shape, n_part=max_n_part_fluid).astype(np.float32)
    n_part_fluid = len(x_fluid)
    rect_shape = sample_rect_shape()
    x_particle = get_particles(rect_shape, n_part=n_part_particle).astype(np.float32)
    x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
    n_particles = n_part_fluid + n_part_particle
    x = ti.Vector.field(2, dtype=float, shape=n_particles)
    x.from_numpy(x_combine)
    fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
    particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)
    fluid.from_numpy(x_fluid)
    particle.from_numpy(x_particle)

    # Initialize up other fields:
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id

    # Reset other fields:
    reset_other_fields(n_particles)
    
    return x, v, C, F, material, Jp, fluid, particle, n_particles, n_part_fluid


# In[ ]:


gravity_amp = 1
max_n_part_fluid = 300
n_part_particle = 300

fluid_shape = sample_shape()
x_fluid = get_fluid(fluid_shape, n_part=max_n_part_fluid).astype(np.float32)
n_part_fluid = len(x_fluid)
print(n_part_fluid)
rect_shape = sample_rect_shape()
x_particle = get_particles(rect_shape, n_part=n_part_particle).astype(np.float32)
x_combine = np.concatenate([x_particle, x_fluid]).astype(np.float32)
n_particles = n_part_fluid + n_part_particle
x = ti.Vector.field(2, dtype=float, shape=n_particles)
x.from_numpy(x_combine)
fluid = ti.Vector.field(2, dtype=float, shape=n_part_fluid)
particle = ti.Vector.field(2, dtype=float, shape=n_part_particle)
fluid.from_numpy(x_fluid)
particle.from_numpy(x_particle)

# Initialize up other fields:
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

# Reset other fields:
reset_other_fields(n_particles)

def main(fluid, particle):
    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )

    res = (512, 512)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
    canvas = window.get_canvas()
    radius = 0.003

    gravity[None] = [0, -1]
    count = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                x, v, C, F, material, Jp, fluid, particle, n_particles, n_part_fluid = reset_all()
            elif window.event.key in [ti.ui.ESCAPE]:
                break
        # if window.event is not None:
        #     gravity[None] = [0, 0]  # if had any event
        # if window.is_pressed(ti.ui.LEFT, 'a'):
        #     gravity[None][0] = -1
        # if window.is_pressed(ti.ui.RIGHT, 'd'):
        #     gravity[None][0] = 1
        # if window.is_pressed(ti.ui.UP, 'w'):
        #     gravity[None][1] = 1
        # if window.is_pressed(ti.ui.DOWN, 's'):
        #     gravity[None][1] = -1
        gravity[None][1] = -gravity_amp
        # mouse = window.get_cursor_pos()
        # mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
        # canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.05)
        # attractor_pos[None] = [mouse[0], mouse[1]]
        # attractor_strength[None] = 0
        # if window.is_pressed(ti.ui.LMB):
        #     attractor_strength[None] = 1
        # if window.is_pressed(ti.ui.RMB):
        #     attractor_strength[None] = -1

        for s in range(int(2e-3 // dt)):
            substep()
        time.sleep(0.1)
        render()
        print(fluid.to_numpy()[0], particle.to_numpy()[0])
        if count in [0, 1, 2]:
            time.sleep(3)
        canvas.set_background_color((0.067, 0.184, 0.255))
        canvas.circles(fluid, radius=radius, color=(0, 0.5, 0.5))
        # # canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
        canvas.circles(particle, radius=radius, color=(0, 0.5, 0.5))
        window.show()
        count += 1


# In[ ]:


if __name__ == '__main__':
    main(fluid, particle)

