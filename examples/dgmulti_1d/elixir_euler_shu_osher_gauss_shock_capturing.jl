using Trixi
using OrdinaryDiffEqSSPRK

gamma_gas = 1.4
equations = CompressibleEulerEquations1D(gamma_gas)

###############################################################################
# setup the GSBP DG discretization that uses the Gauss operators from
# Chan, Del Rey Fernandez, Carpenter (2019).
# [https://doi.org/10.1137/18M1209234](https://doi.org/10.1137/18M1209234)

# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# [https://doi.org/10.1016/0021-9991(89)90222-2](https://doi.org/10.1016/0021-9991(89)90222-2)
function initial_condition_shu_osher(x, t, equations::CompressibleEulerEquations1D)
    x0 = -4

    rho_left = 27 / 7
    v_left = 4 * sqrt(35) / 9
    p_left = 31 / 3

    v_right = 0.0
    p_right = 1.0

    rho = ifelse(x[1] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x[1] > x0, v_right, v_left)
    p = ifelse(x[1] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p),
                     equations)
end

initial_condition = initial_condition_shu_osher

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Line(), polydeg, approximation_type = GaussSBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :entire_boundary => boundary_condition)

###############################################################################
#  setup the 1D mesh

cells_per_dimension = (64,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-5.0,), coordinates_max = (5.0,),
                   periodicity = false)

###############################################################################
#  setup the semidiscretization and ODE problem

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    dg, boundary_conditions = boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

###############################################################################
#  setup the callbacks

# prints a summary of the simulation setup and resets the timers
summary_callback = SummaryCallback()

# analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

# SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.2)

# collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

# ###############################################################################
# # run the simulation

sol = solve(ode, SSPRK43(), adaptive = false;
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            callback = callbacks, ode_default_options()...)
