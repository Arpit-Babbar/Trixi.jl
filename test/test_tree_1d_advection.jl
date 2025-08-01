module TestExamples1DAdvection

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[6.0388296447998465e-6],
                        linf=[3.217887726258972e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_basic.jl (max_abs_speed)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        surface_flux=FluxLaxFriedrichs(max_abs_speed),
                        l2=[6.0388296447998465e-6],
                        linf=[3.217887726258972e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        l2=[0.3540206249507417],
                        linf=[0.9999896603382347])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
                        l2=[4.283508859843524e-6],
                        linf=[3.235356127918171e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_basic.jl (No errors)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        analysis_callback=AnalysisCallback(semi, interval = 42,
                                                           analysis_errors = Symbol[]))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_finite_volume.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_finite_volume.jl"),
                        l2=[0.011662300515980219],
                        linf=[0.01647256923710194])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_perk2.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2.jl"),
                        l2=[0.011288030389423475],
                        linf=[0.01596735472556976])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom 
        # integrator which are not *recorded* for the methods from 
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
    end
end

# Testing the second-order paired explicit Runge-Kutta (PERK) method without stepsize callback
@trixi_testset "elixir_advection_perk2.jl(fixed time step)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2.jl"),
                        dt=2.0e-3,
                        tspan=(0.0, 20.0),
                        save_solution=SaveSolutionCallback(dt = 0.1 + 1.0e-8),
                        callbacks=CallbackSet(summary_callback, save_solution,
                                              analysis_callback, alive_callback),
                        l2=[9.886271430207691e-6],
                        linf=[3.729460413781638e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom 
        # integrator which are not *recorded* for the methods from 
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
    end
end

# Testing the second-order paired explicit Runge-Kutta (PERK) method with the optimal CFL number
@trixi_testset "elixir_advection_perk2_optimal_cfl.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2_optimal_cfl.jl"),
                        l2=[0.0009700887119146429],
                        linf=[0.00137209242077041])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom 
        # integrator which are not *recorded* for the methods from 
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
    end
end

@trixi_testset "elixir_advection_doublefloat.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_doublefloat.jl"),
                        l2=Double64[6.80895929885700039832943251427357703e-11],
                        linf=Double64[5.82834770064525291688100323411704252e-10])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_uncertainty.jl" begin
    # Run this first to collect LoopVectorization warnings before error-check run
    @test_nowarn_mod trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                            "elixir_advection_uncertainty.jl"))

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_uncertainty.jl"),
                        l2=[0.0012576893000440965 ± 0.017581020765034417],
                        linf=[0.004425204509676317 ± 0.0633672486044246])
    # Using Measurements.jl actually allocates quite significantly
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 100_000
    end
end
end

end # module
