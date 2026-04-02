import jax
import jax.numpy as jnp

#Example system
def double_pendulum_dynamics(x, u):
    theta1, theta2, dtheta1, dtheta2 = x[0], x[1], x[2], x[3]
    tau1, tau2 = u[0], u[1]
    m1, m2, l1, l2 = 2.0, 1.0, 1.0, 2.0
    g = 10.0
    #System dynamics
    d_theta1_dt = dtheta1
    d_theta2_dt = dtheta2
    dd_theta1_dt = jnp.sin(theta1) + tau1
    dd_theta2_dt = jnp.sin(theta2) + tau2
    #Return xdot
    return jnp.array([d_theta1_dt, d_theta2_dt, dd_theta1_dt, dd_theta2_dt])

def build_taylor_graph(system_fn, max_degree):
    #Dictionary to store generated derivative functions
    deriv_funcs = {0: {'': system_fn}}
    for deg in range(1, max_degree + 1):
        deriv_funcs[deg] = {}
        for key, prev_fn in deriv_funcs[deg - 1].items():
            deriv_funcs[deg][key + 'u'] = jax.jacfwd(prev_fn, argnums=1)
            if 'u' not in key:
                deriv_funcs[deg][key + 'x'] = jax.jacfwd(prev_fn, argnums=0)

    @jax.jit
    def compute_taylor_coeffs(x_op, u_op):
        coeffs = []
        #Degree 0
        coeffs.append(deriv_funcs[0][''](x_op, u_op))
        #Higher degrees
        for deg in range(1, max_degree + 1):
            deg_coeffs = {}
            #Sort keys
            sorted_keys = sorted(deriv_funcs[deg].keys(), key=lambda k: (-k.count('x'), k))
            for k in sorted_keys:
                deg_coeffs[k] = deriv_funcs[deg][k](x_op, u_op)
            coeffs.append(deg_coeffs)
        return coeffs
    return compute_taylor_coeffs

#Usage Example:
#Define JAX arrays for operating point
x_eq = jnp.array([0.0, 0.0, 0.0, 0.0])
u_eq = jnp.array([0.0, 0.0])
graph = build_taylor_graph(double_pendulum_dynamics, 2)
values = graph(x_eq, u_eq)
print(values)
import pdb; pdb.set_trace()
