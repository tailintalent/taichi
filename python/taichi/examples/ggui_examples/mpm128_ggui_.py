import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

quality = 4  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-5 / quality
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

materials = [9000*(quality**2 - 2), 9000, 9000]

# material_indices = []
# for i in range(len(materials)):
#     material_indices += [i] * materials[i]

water = ti.Vector.field(2, dtype=float, shape=materials[0])  # position
jelly = ti.Vector.field(2, dtype=float, shape=materials[1])  # position
snow = ti.Vector.field(2, dtype=float, shape=materials[2])  # position
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0 or material[p] == 1:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            new_sig = min(max(sig[d, d], .99 - 4.5e-2),.99 - 3.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        # if material[p] == 0:
        #     # Reset deformation gradient to avoid numerical instability
        #     F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        # elif material[p] == 2:
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
    gravity[None] = [0, -100]
    for i in range(n_particles):  
        if i < 9000*(quality**2 - 2):
            material[i] = 0 
        elif i < 9000*(quality**2 - 1):
            material[i] = 1
        else: 
            material[i] = 2
        # 0: fluid 1: jelly 2: snow

        if material[i] == 0:
            x[i] = [
            ti.random(),
            ti.random() * 0.1
            ]
            v[i] = [0, 0]
        elif material[i] == 1:
            x[i] = [
            0.1 * ti.random() + 0.7,
            ti.random() * 0.1 + 0.9
            ]
            v[i] = [0, 0]
        elif material[i] == 2:
            x[i] = [
            ti.random(),
            ti.random() * 0.5 + 0.5
            ]
            v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def render():
    for i in range(9000*(quality**2 - 2)):
        water[i] = x[i]
    for i in range(9000*(quality**2 - 2),9000*(quality**2 - 1)):
        jelly[i] = x[i]
    for i in range(9000*(quality**2 - 1), 9000*(quality**2)):
        snow[i] = x[i]



def main():
    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )

    res = (512, 512)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
    canvas = window.get_canvas()
    radius = 0.003

    reset()


    while window.running:
        # gravity[None] = [0, 0]
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                reset()
            elif window.event.key in [ti.ui.ESCAPE]:
                break
        gravity[None][1] = -50
        # if window.event is not None:
        #     gravity[None] = [0, 0]  # if had any event
        # if window.is_pressed(ti.ui.LEFT, 'a'):
        #     gravity[None][0] = -1
        #     gravity[None][1] = -100
        # if window.is_pressed(ti.ui.RIGHT, 'd'):
        #     gravity[None][0] = 1
        #     gravity[None][1] = -10
        # if window.is_pressed(ti.ui.UP, 'w'):
        #     gravity[None][1] = 1
        # if window.is_pressed(ti.ui.DOWN, 's'):
            # gravity[None][1] = -100
        mouse = window.get_cursor_pos()
        mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
        canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.05)
        attractor_pos[None] = [mouse[0], mouse[1]]
        attractor_strength[None] = 0
        if window.is_pressed(ti.ui.LMB):
            attractor_strength[None] = 1
        if window.is_pressed(ti.ui.RMB):
            attractor_strength[None] = -1

        for s in range(int(2e-3 // dt)):
            substep()
        render()
        canvas.set_background_color((0.067, 0.184, 0.255))
        canvas.circles(water, radius=radius, color=(0, 0.5, 0.5))
        canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
        canvas.circles(snow, radius=radius, color=(1, 1, 1))
        window.show()


if __name__ == '__main__':
    main()
