using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 2.0
kappa = 0.5     # Scaling factor for the pressure.
epsilon = 0.1
equations = PolytropicEulerEquationsPerturbed2D(gamma, kappa, epsilon)
equations = PolytropicEulerEquations2D(gamma, kappa)

initial_condition = Trixi.initial_condition_isentropic_vortex

volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs,
            #    volume_integral = VolumeIntegralFluxDifferencing(flux_central)
               )

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.1)

visualization = VisualizationCallback(interval = 200)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, visualization,
                        stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary


equations_stiff = PolytropicEulerEquationsStiff2D(gamma, kappa, epsilon)
equations_nonstiff = PolytropicEulerEquationsNonStiff2D(gamma, kappa, epsilon)

volume_flux = flux_winters_etal
solver_central = DGSEM(polydeg = 0, surface_flux = volume_flux,
            #    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
               )

semi_stiff = SemidiscretizationHyperbolic(mesh, equations_stiff, initial_condition, solver_central)

semi_nonstiff = SemidiscretizationHyperbolic(mesh, equations_nonstiff, initial_condition, solver)

implicit_solver_imex = IMEX111()

sol_nonstiff = deepcopy(sol)

# sol_implicit = Trixi.implicit_solve(implicit_solver_imex, deepcopy(sol_nonstiff), stepsize_callback, tspan, semi_stiff, semi_nonstiff)

analysis_callback(sol)
analysis_callback(sol_implicit)

plot(sol_implicit)
plot(sol)
