module TestExamples2DAdvection

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_extended.jl with polydeg=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[0.02134571266411136],
                        linf=[0.04347734797775926],
                        polydeg=1)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart.jl" begin
    using OrdinaryDiffEqSSPRK: SSPRK43
    println("═"^100)
    println(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration_adaptive.jl"))
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_advection_timeintegration_adaptive.jl"),
                  alg = SSPRK43(), tspan = (0.0, 10.0))
    l2_expected, linf_expected = analysis_callback(sol)

    println("═"^100)
    println(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"))
    # Errors are exactly the same as in the elixir_advection_extended.jl
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                  alg = SSPRK43(),
                  base_elixir = "elixir_advection_timeintegration_adaptive.jl")
    l2_actual, linf_actual = analysis_callback(sol)

    @test l2_actual == l2_expected
    @test linf_actual == linf_expected
end

@trixi_testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[0.0015188466707237375],
                        linf=[0.008446655719187679])

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
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[4.913300828257469e-5],
                        linf=[0.00045263895394385967])
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
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[3.2207388565869075e-5],
                        linf=[0.0007508059772436404])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_solution_independent.jl"),
                        l2=[4.949660644033807e-5],
                        linf=[0.0004867846262313763])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_visualization.jl" begin
    # To make CI tests work, disable showing a plot window with the GR backend of the Plots package
    # Xref: https://github.com/jheinen/GR.jl/issues/278
    # Xref: https://github.com/JuliaPlots/Plots.jl/blob/8cc6d9d48755ba452a2835f9b89d3880e9945377/test/runtests.jl#L103
    if !isinteractive()
        restore = get(ENV, "GKSwstype", nothing)
        ENV["GKSwstype"] = "100"
    end

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_visualization.jl"),
                        l2=[0.0007225529919720868],
                        linf=[0.005954447875428925])

    # Restore GKSwstype to previous value (if it was set)
    if !isinteractive()
        if isnothing(restore)
            delete!(ENV, "GKSwstype")
        else
            ENV["GKSwstype"] = restore
        end
    end
end

@trixi_testset "elixir_advection_timeintegration.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[2.4976030518356626e-5],
                        linf=[0.0005531580316338533])
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end
end

@trixi_testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[2.5314747030031457e-5],
                        linf=[0.0005437136621948904],
                        ode_algorithm=Trixi.CarpenterKennedy2N43(),
                        cfl=1.0)
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43 with maxiters=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[1.2135350502911197e-5],
                        linf=[9.999985420537649e-5],
                        ode_algorithm=Trixi.CarpenterKennedy2N43(),
                        cfl=1.0,
                        maxiters=1)
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk94" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[2.4976673477385313e-5],
                        linf=[0.0005534166916640881],
                        ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar94())
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[3.667894656471403e-5],
                        linf=[0.0005799465470165757],
                        ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32(),
                        cfl=1.0)
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32 with maxiters=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
                        l2=[1.2198725469737875e-5],
                        linf=[9.977247740793407e-5],
                        ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32(),
                        cfl=1.0,
                        maxiters=1)
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_advection_callbacks.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_callbacks.jl"),
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

# Coverage test for all initial conditions
@testset "Linear scalar advection: Tests for initial conditions" begin
    # Linear scalar advection
    @trixi_testset "elixir_advection_extended.jl with initial_condition_sin_sin" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                            l2=[0.0001420618061089383],
                            linf=[0.0007140570281718439],
                            maxiters=1,
                            initial_condition=Trixi.initial_condition_sin_sin)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_extended.jl with initial_condition_constant" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                            l2=[3.8302867746057483e-16],
                            linf=[1.3322676295501878e-15],
                            maxiters=1,
                            initial_condition=initial_condition_constant)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_x_y" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                            l2=[2.7276160570381226e-16],
                            linf=[5.10702591327572e-15],
                            maxiters=1,
                            initial_condition=Trixi.initial_condition_linear_x_y,
                            boundary_conditions=Trixi.boundary_condition_linear_x_y,
                            periodicity=false)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_x" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                            l2=[1.5121648229368207e-16],
                            linf=[1.3322676295501878e-15],
                            maxiters=1,
                            initial_condition=Trixi.initial_condition_linear_x,
                            boundary_conditions=Trixi.boundary_condition_linear_x,
                            periodicity=false)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_y" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                            l2=[1.714292614252588e-16],
                            linf=[2.220446049250313e-15],
                            maxiters=1,
                            initial_condition=Trixi.initial_condition_linear_y,
                            boundary_conditions=Trixi.boundary_condition_linear_y,
                            periodicity=false)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
end

end # module
