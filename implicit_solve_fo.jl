using TrixiBase: trixi_include
using NLsolve
import OrdinaryDiffEq: solve

struct ImplicitSolve
end

# First you can do this dry run to get the `sol` object on which you can use callbacks
# like analysis_callback(sol)

# sol = trixi_include("examples/tree_2d_dgsem/elixir_euler_polytropic_ec.jl",
#                     dt = 1e-5)

sol = trixi_include("examples/tree_2d_dgsem/elixir_euler_density_wave.jl",
                    initial_refinement_level = 2);

analysis_callback(sol)

du_imp = similar(sol.u[1]) # du will be needed for the rhs! function
u_old = deepcopy(sol.u[1]) # u_old will be needed for the implicit solver

dt = 1e-5

function apply_rhs_imp!(F, du, u, u_new, semi::SemidiscretizationHyperbolic, dt, t)
    # Apply the rhs! function

    rhs!(du, u_new, semi, t)
    # Update the u_ode

    # The solution to the equation will be the zero of
    # u^{n+1} - (u^n + dt * u^{n+1})
    for i in eachindex(F)
        F[i] = u_new[i] - (u[i] + dt * du[i])
    end

    return nothing
end

function generate_function_of_F_y(du_ode, u_ode, semi::SemidiscretizationHyperbolic, dt, t, func)
    function function_of_y(F, y)
        return func(F, du_ode, u_ode, y, semi, dt, t)
    end
end

F = generate_function_of_F_y(du_imp, u_old, semi, dt, dt, apply_rhs_imp!)

# Solver with Picard's iterative solver (TODO - Try AD by replace m = 0 with autodiff = :forward)
sol_nl = nlsolve(F, sol.u[1], autodiff = :finite, xtol = 1e-13, ftol = 1e-14)
@show norm(sol_nl.zero .- sol.u[2], 2)


# Extra

function apply_rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, dt, t)
    # Apply the rhs! function
    rhs!(du_ode, u_ode, semi, t)

    # Update the u_ode

    u_ode .+= dt * du_ode

    return u_ode
end

dt = stepsize_callback(ode) # Time step can be obtained like this

function solve(::ImplicitSolve, sol, stepsize_callback, tspan, semi)
    t = 0.0
    Tf = sol.t[2]

    u_new = sol.u[2]
    u_old = copy(u_new)
    du = similar(u_old)

    while t < Tf
        dt = stepsize_callback(sol.prob)

        if t + dt > Tf
            dt = Tf - t
        end

        F = generate_function_of_F_y(du, u_old, semi, dt, t, apply_rhs_imp!)
        sol_nl = nlsolve(F, u_old, autodiff = :finite, xtol = 1e-12, ftol = 1e-12)
        @show sol_nl
        @assert sol_nl.x_converged || sol_nl.f_converged sol_nl
        u_new .= sol_nl.zero
        u_old .= u_new

        @show t,dt
        t += dt
    end
    @assert t == Tf

    return sol
end

implicit_solve = ImplicitSolve()
backup = deepcopy(sol.u[2])
solve(implicit_solve, sol, stepsize_callback, tspan, semi);
analysis_callback(sol)

analysis_callback(sol)

# The IMEX one will simply be implemented as

function apply_rhs_imp!(F, du_1, du_2, u_old, u_new, semi_1::SemidiscretizationHyperbolic,
                        semi_2::SemidiscretizationHyperbolic, dt, t)
    # Apply the rhs! function

    rhs!(du_1, u_new, semi_1, t) # Implicit part
    rhs!(du_2, u_old, semi_2, t) # Explicit part
    # Update the u_ode

    F .= u_new .- (u_old .+ dt * (du_1 .+ du_2))

    return nothing
end


# 0.03996654929714758, 0.07920068572936301, 0.09736881674073727], linf = [0.08555257090680612, 0.15572328032495708, 0.20551785156506952