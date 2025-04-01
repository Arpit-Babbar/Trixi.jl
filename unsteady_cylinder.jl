# Channel flow around a cylinder at Mach 3
#
# Boundary conditions are supersonic Mach 3 inflow at the left portion of the domain
# and supersonic outflow at the right portion of the domain. The top and bottom of the
# channel as well as the cylinder are treated as Euler slip wall boundaries.
# This flow results in strong shock reflections / interactions as well as Kelvin-Helmholtz
# instabilities at later times as two Mach stems form above and below the cylinder.
#
# For complete details on the problem setup see Section 5.7 of the paper:
# - Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, and Ignacio Tomas (2018)
#   Second-Order Invariant Domain Preserving Approximation of the Euler Equations using Convex Limiting.
#   [DOI: 10.1137/17M1149961](https://doi.org/10.1137/17M1149961)
#
# Keywords: supersonic flow, shock capturing, AMR, unstructured curved mesh, positivity preservation, compressible Euler, 2D

using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 1.5
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_flow(x, t, equations)
    flux = Trixi.flux(u_boundary, normal_direction, equations)

    return flux
end

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = Dict(:Bottom => boundary_condition_outflow,
                           :Cylinder => boundary_condition_slip_wall,
                           :Top => boundary_condition_outflow,
                           :Right => boundary_condition_outflow,
                           :Left => boundary_condition_supersonic_inflow)
volume_flux = flux_ranocha_turbo
surface_flux = flux_lax_friedrichs

polydeg = 6
basis = LobattoLegendreBasis(polydeg)
# `density_pressure` is basically rho*p.
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
# The following is the hybrid method for capturing the shock. The
# flux used for the lower order finite volume scheme is the surface_flux itself,
# whereas for the DG method is the volume_flux.
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = joinpath("cylinder.inp")

mesh = P4estMesh{2}(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "./results",
                                     analysis_filename="analysis_mach_0150.dat")

alive_callback = AliveCallback(analysis_interval = analysis_interval)

#Save the current numerical solution in regular intervals.
save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = "mach_0150/",
                                     solution_variables = cons2prim)

# This indicator just computes a weighted second derivative of the density.
amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

# The threshold determines that the associated level of refinement is done for
# indictor values above the refinement.
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 3, med_threshold = 0.05,
                                      max_level = 5, max_threshold = 0.1)

# Below interval determines the number of time steps after which AMR should
# be performed. adapt_initial_condition indicates whether the initial condition
# already should be adapted before the first time step. And with
# adapt_initial_condition_only_refine=true the mesh is only refined
#at the beginning but not coarsened.
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 3,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        amr_callback
                        )

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure, Trixi.density))

###############################################################################
# run the simulation
# Here, we use an adaptive time-step integration scheme. (adaptive can be set to false)
sol = solve(ode,
            # SSPRK43(stage_limiter!, OrdinaryDiffEqLowStorageRK.trivial_limiter!, OrdinaryDiffEqSSPRK.DiffEqBase.False());
            CarpenterKennedy2N54(stage_limiter!, OrdinaryDiffEqLowStorageRK.trivial_limiter!, OrdinaryDiffEqSSPRK.DiffEqBase.True(), false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
