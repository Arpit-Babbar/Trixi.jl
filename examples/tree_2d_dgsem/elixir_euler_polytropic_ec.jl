
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 1.4
kappa = 0.5     # Scaling factor for the pressure.
epsilon = 0.001
equations = PolytropicEulerEquationsPerturbed2D(gamma, kappa, epsilon)

initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

surface_and_volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 0, surface_flux = surface_and_volume_flux,
               volume_integral = VolumeIntegralFluxDifferencing(surface_and_volume_flux))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = true,
                n_cells_max = 30_000)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.002)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 0.05)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            Euler(),
            # CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
