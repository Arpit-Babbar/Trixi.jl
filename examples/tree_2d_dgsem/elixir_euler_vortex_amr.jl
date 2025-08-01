using OrdinaryDiffEqLowStorageRK
using Trixi

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension

using Trixi

struct IndicatorVortex{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorVortex(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    A = Array{real(basis), 2}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    cache = (; semi.mesh, alpha, indicator_threaded)

    return IndicatorVortex{typeof(cache)}(cache)
end

function (indicator_vortex::IndicatorVortex)(u::AbstractArray{<:Any, 4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
    mesh = indicator_vortex.cache.mesh
    alpha = indicator_vortex.cache.alpha
    resize!(alpha, nelements(dg, cache))

    # get analytical vortex center (based on assumption that center=[0.0, 0.0]
    # at t=0.0 and that we stop after one period)
    domain_length = mesh.tree.length_level_0
    if t < 0.5f0 * domain_length
        center = (t, t)
    else
        center = (t - domain_length, t - domain_length)
    end

    Threads.@threads for element in eachelement(dg, cache)
        cell_id = cache.elements.cell_ids[element]
        coordinates = (mesh.tree.coordinates[1, cell_id], mesh.tree.coordinates[2, cell_id])
        # use the negative radius as indicator since the AMR controller increases
        # the level with increasing value of the indicator and we want to use
        # high levels near the vortex center
        alpha[element] = -periodic_distance_2d(coordinates, center, domain_length)
    end

    return alpha
end

function periodic_distance_2d(coordinates, center, domain_length)
    dx = @. abs(coordinates - center)
    dx_periodic = @. min(dx, domain_length - dx)
    return sqrt(sum(abs2, dx_periodic))
end

# Optional: Nicer display of the indicator
function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorVortex)
    Trixi.summary_box(io, "IndicatorVortex")
end

end # module TrixiExtension

import .TrixiExtension

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case of
- Chi-Wang Shu (1997)
  Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
  Schemes for Hyperbolic Conservation Laws
  [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
    # for error convergence: make sure that the end time is such that the vortex is back at the initial state!!
    # for the current velocity and domain size: t_end should be a multiple of 20s
    # initial center of the vortex
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    # size and strength of the vortex
    iniamplitude = 5
    # base flow
    rho = 1
    v1 = 1
    v2 = 1
    vel = SVector(v1, v2)
    p = convert(RealT, 25)
    rt = p / rho                  # ideal gas equation
    t_loc = 0
    cent = inicenter + vel * t_loc      # advection of center
    # ATTENTION: handle periodic BC, but only for v1 = v2 = 1.0 (!!!!)

    cent = x - cent # distance to center point

    # cent = cross(iniaxis, cent) # distance to axis, tangent vector, length r
    # cross product with iniaxis = [0, 0, 1]
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude / (2 * convert(RealT, pi)) * exp(0.5f0 * (1 - r2)) # vel. perturbation
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2 # isentropic
    rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex
# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

coordinates_min = (-10.0, -10.0)
coordinates_max = (10.0, 10.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (entropy, energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Add `:temperature` to `extra_node_variables` tuple ...
extra_node_variables = (:temperature,)

# ... and specify the function `get_node_variable` for this symbol, 
# with first argument matching the symbol (turned into a type via `Val`) for dispatching.
function Trixi.get_node_variable(::Val{:temperature}, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    temp_array = zeros(eltype(cache.elements),
                       n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                       n_elements)

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            rho, _, _, p = prim2cons(u_node, equations)
            temp = p / rho # ideal gas equation with R = 1

            temp_array[i, j, element] = temp
        end
    end

    return temp_array
end
save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = extra_node_variables) # Supply the additional `extra_node_variables` here

amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorVortex(semi),
                                      base_level = 3,
                                      med_level = 4, med_threshold = -3.0,
                                      max_level = 5, max_threshold = -2.0)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
