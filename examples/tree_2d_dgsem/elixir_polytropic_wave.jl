using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 2.0   # Adiabatic monatomic gas in 2d.
kappa = 0.5   # Scaling factor for the pressure.
epsilon = 0.1
equations = PolytropicEulerEquationsPerturbed2D(gamma, kappa, epsilon)
# equations = PolytropicEulerEquations2D(gamma, kappa)

# Linear pressure wave in the negative x-direction.
function initial_condition_wave(x, t, equations::PolytropicEulerEquationsPerturbed2D)
    rho = 1.0
    v1 = 0.0
    if x[1] > 0.0
        rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)) / equations.kappa)^(1 / equations.gamma)
        v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)) / equations.kappa)
    end
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end
initial_condition = initial_condition_wave

volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 0, surface_flux = flux_lax_friedrichs,
            #    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
               )

coordinates_min = (-2.0, -1.0)
coordinates_max = (2.0, 1.0)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                periodicity = true,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

visualization = VisualizationCallback(interval = 1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        visualization,
                        stepsize_callback
                        )

###############################################################################
# run the simulation

sol = solve(ode,
            # CarpenterKennedy2N54(williamson_condition = false),
            Euler(),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

implicit_solver = Trixi.FullyImplicitSolver()

tspan = (0.0, 1.0)

equations_stiff = PolytropicEulerEquationsStiff2D(gamma, kappa, epsilon)
equations_nonstiff = PolytropicEulerEquationsNonStiff2D(gamma, kappa, epsilon)

volume_flux = flux_winters_etal
solver_central = DGSEM(polydeg = 0, surface_flux = volume_flux,
            #    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
               )

semi_stiff = SemidiscretizationHyperbolic(mesh, equations_stiff, initial_condition, solver_central,
                                          boundary_conditions = boundary_conditions)

semi_nonstiff = SemidiscretizationHyperbolic(mesh, equations_nonstiff, initial_condition, solver,
                                             boundary_conditions = boundary_conditions)

implicit_solver_imex = IMEX111()

# sol_nonstiff = deepcopy(sol)

sol_implicit = Trixi.implicit_solve(implicit_solver_imex, deepcopy(sol_nonstiff), stepsize_callback, tspan, semi_stiff, semi_nonstiff)

analysis_callback(sol)
analysis_callback(sol_implicit)

plot(sol_implicit)

# Print the timer summary
# summary_callback()
