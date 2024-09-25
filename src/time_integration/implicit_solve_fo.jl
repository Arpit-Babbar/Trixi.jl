using NLsolve

struct FullyImplicitSolver
end

struct IMEX111
end

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

function apply_rhs_imex111!(F, du_stiff, du_nonstiff, u_old, u_new,
                            semi_stiff::SemidiscretizationHyperbolic,
                            semi_nonstiff::SemidiscretizationHyperbolic, dt, t)

    # Apply the rhs! function
    rhs!(du_stiff, u_new, semi_stiff, t)
    rhs!(du_nonstiff, u_old, semi_nonstiff, t)

    # Update the u_ode

    # The solution to the equation will be the zero of
    # u^{n+1} - (u^n + dt * u^{n+1})
    for i in eachindex(F)
        F[i] = u_new[i] - (u_old[i] + dt * (du_stiff[i] + du_nonstiff[i]))
    end

    return nothing
end

function generate_function_of_F_y(du_ode, u_ode, semi::SemidiscretizationHyperbolic, dt, t, func)
    function function_of_y(F, y)
        return func(F, du_ode, u_ode, y, semi, dt, t)
    end
end

function generate_function_of_F_y(du_stiff, du_nonstiff, u_old,
                                          semi_stiff::SemidiscretizationHyperbolic,
                                          semi_nonstiff::SemidiscretizationHyperbolic,
                                          dt, t, func)
    function function_of_y(F, y)
        return func(F, du_stiff, du_nonstiff, u_old, y, semi_stiff, semi_nonstiff, dt, t)
    end
end

function implicit_solve(::FullyImplicitSolver, sol, stepsize_callback, tspan, semi)
    t = 0.0
    Tf = tspan[2]

    u_new = sol.u[2]
    u_old = copy(u_new)
    du = similar(u_old)

    while t < Tf
        dt = stepsize_callback(sol.prob)

        if t + dt > Tf
            dt = Tf - t
        end

        F = generate_function_of_F_y(du, u_old, semi, dt, t, apply_rhs_imp!)
        sol_nl = nlsolve(F, u_old, m=0, xtol = 1e-12, ftol = 1e-12)
        # @show sol_nl
        @assert sol_nl.x_converged || sol_nl.f_converged sol_nl
        u_new .= sol_nl.zero
        u_old .= u_new

        @show t,dt
        t += dt
    end
    @assert t == Tf

    return sol
end

function implicit_solve(::IMEX111, sol, stepsize_callback, tspan,
                        semi_stiff::SemidiscretizationHyperbolic,
                        semi_nonstiff::SemidiscretizationHyperbolic)
    t = 0.0
    Tf = tspan[2]

    u_new = sol.u[2]
    u_old = copy(u_new)
    du_stiff = similar(u_old)
    du_nonstiff = similar(u_old)

    while t < Tf
        dt = stepsize_callback(sol.prob)

        if t + dt > Tf
            dt = Tf - t
        end

        F = generate_function_of_F_y(du_stiff, du_nonstiff, u_old, semi_stiff, semi_nonstiff,
                                     dt, t, apply_rhs_imex111!)
        sol_nl = nlsolve(F, u_old, m=0, xtol = 1e-12, ftol = 1e-12)
        # @show sol_nl
        @assert sol_nl.x_converged || sol_nl.f_converged sol_nl
        u_new .= sol_nl.zero
        u_old .= u_new

        @show t,dt
        t += dt
    end
    @assert t == Tf

    return sol
end

