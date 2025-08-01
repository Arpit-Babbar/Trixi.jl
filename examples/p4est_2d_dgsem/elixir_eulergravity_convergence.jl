using OrdinaryDiffEqLowStorageRK
using Trixi

initial_condition = initial_condition_eoc_test_coupled_euler_gravity

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 2.0
equations_euler = CompressibleEulerEquations2D(gamma)

polydeg = 3
solver_euler = DGSEM(polydeg, FluxHLL(min_max_speed_naive))

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)

trees_per_dimension = (1, 1)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 2)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler,
                                          source_terms = source_terms_eoc_test_coupled_euler_gravity)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver_gravity = DGSEM(polydeg, FluxLaxFriedrichs(max_abs_speed_naive))

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            source_terms = source_terms_harmonic)

###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density = 2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant = 1.0, # aka G
                                    cfl = 1.1,
                                    resid_tol = 1.0e-10,
                                    n_iterations_max = 1000,
                                    timestep_gravity = timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl = 0.8)

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true)

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
