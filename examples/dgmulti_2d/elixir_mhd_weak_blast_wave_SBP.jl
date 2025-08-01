using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

# For approximation_type = SBP(), Github CI results differ from those attained locally.
# See https://github.com/trixi-framework/Trixi.jl/issues/881.
cells_per_dimension = (16, 16)
c_h = maximum(cells_per_dimension)
equations = IdealGlmMhdEquations2D(1.4, c_h)

initial_condition = initial_condition_weak_blast_wave

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-2.0, -2.0), coordinates_max = (2.0, 2.0),
                   periodicity = true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# See comment above and https://github.com/trixi-framework/Trixi.jl/issues/881
# DGMulti uses a conservative timestep estimate, so we can use a large CFL here.
# cfl = 1.0
# stepsize_callback = StepsizeCallback(cfl=cfl)
#
# glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        #stepsize_callback,
                        alive_callback, save_solution)                        #=glm_speed_callback=#

###############################################################################
# run the simulation

# See comment above and https://github.com/trixi-framework/Trixi.jl/issues/881
# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks);
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks)
