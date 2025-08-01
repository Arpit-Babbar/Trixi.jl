module TestExamplesP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "P4estMesh2D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
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
    semi32 = Trixi.trixi_adapt(Array, Float32, semi)
    @test real(semi32.solver) == Float32
    @test real(semi32.solver.basis) == Float32
    @test real(semi32.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
    @test real(semi32.mesh) == Float64
end

@trixi_testset "elixir_advection_basic.jl (Float32)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test_broken (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64
end

@trixi_testset "elixir_advection_nonconforming_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonconforming_flag.jl"),
                        l2=[3.198940059144588e-5],
                        linf=[0.00030636069494005547])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
                        l2=[0.0005379687442422346],
                        linf=[0.007438525029884735])
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
                        # Expected errors are exactly the same as with StructuredMesh!
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

@trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_flag.jl"),
                        l2=[0.0012808538770535593],
                        linf=[0.01752690016659812])
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
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[4.507575525876275e-6],
                        linf=[6.21489667023134e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_amr.jl"),
                        l2=[2.869137983727866e-6],
                        linf=[3.8353423270964804e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_meshview.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_meshview.jl"),
                        l2=[0.00013773915040249946],
                        linf=[0.0010140184322192658])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end

    # Ensure we cover the calculation of the node coordinates
    node_coordinates = typeof(parent_mesh.tree_node_coordinates)(undef, 2,
                                                                 ntuple(_ -> length(parent_mesh.nodes),
                                                                        2)...,
                                                                 length(mesh.cell_ids))
    result = Trixi.calc_node_coordinates!(node_coordinates, mesh, parent_mesh.nodes)
    @test parent_mesh.tree_node_coordinates == result

    # Load the mesh file for code coverage.
    loaded_mesh = Trixi.load_mesh_serial(joinpath("out", "mesh.h5"); n_cells_max = 0,
                                         RealT = typeof(parent_mesh).parameters[3])
end

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with P4estMeshView!
                        l2=[0.00013773915040249946],
                        linf=[0.0010140184322192658],
                        initial_refinement_level=0)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                        l2=[
                            0.0034516244508588046,
                            0.0023420334036925493,
                            0.0024261923964557187,
                            0.004731710454271893
                        ],
                        linf=[
                            0.04155789011775046,
                            0.024772109862748914,
                            0.03759938693042297,
                            0.08039824959535657
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.063350241405049e-15,
                            1.8571016296925367e-14,
                            3.1769447886391905e-14,
                            1.4104095258528071e-14
                        ],
                        linf=[1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
                        atol=2.0e-12,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream_hybrid_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_hybrid_mesh.jl"),
                        l2=[
                            1.0174922714929637e-15,
                            5.053352600778435e-15,
                            7.358169131303026e-15,
                            5.999843977180112e-15
                        ],
                        linf=[
                            4.440892098500626e-15,
                            2.6117996654306808e-14,
                            4.246603069191224e-14,
                            5.861977570020827e-14
                        ],
                        atol=2.0e-12,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                        l2=[
                            9.53984675e-02,
                            1.05633455e-01,
                            1.05636158e-01,
                            3.50747237e-01
                        ],
                        linf=[
                            2.94357464e-01,
                            4.07893014e-01,
                            3.97334516e-01,
                            1.08142520e+00
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec.jl (flux_chandrashekar)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                        l2=[
                            0.09527896382082567,
                            0.10557894830184737,
                            0.10559379376154387,
                            0.3503791205165925
                        ],
                        linf=[
                            0.2733486454092644,
                            0.3877283966722886,
                            0.38650482703821426,
                            1.0053712251056308
                        ],
                        tspan=(0.0, 1.0),
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec_float32.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_shockcapturing_ec_float32.jl"),
                        l2=[
                            0.09539953f0,
                            0.10563527f0,
                            0.105637245f0,
                            0.3507514f0
                        ],
                        linf=[
                            0.2930063f0,
                            0.4079147f0,
                            0.3972956f0,
                            1.0764117f0
                        ],
                        tspan=(0.0f0, 1.0f0),
                        rtol=10 * sqrt(eps(Float32)), # to make CI pass
                        RealT=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            3.76149952e-01,
                            2.46970327e-01,
                            2.46970327e-01,
                            1.28889042e+00
                        ],
                        linf=[
                            1.22139001e+00,
                            1.17742626e+00,
                            1.17742626e+00,
                            6.20638482e+00
                        ],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
                        l2=[
                            0.4573787784168518,
                            0.28520972760728397,
                            0.28527281808006966,
                            1.2881460122982442
                        ],
                        linf=[
                            1.644411040701827,
                            1.6743368119653912,
                            1.6760847977977988,
                            6.268843623142863
                        ],
                        tspan=(0.0, 0.3))
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

    # Test `resize!`
    ode_alg = Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks)
    integrator = Trixi.init(ode, ode_alg, dt = 42.0, callback = callbacks)

    resize!(integrator, 42)
    @test length(integrator.u) == 42
    @test length(integrator.du) == 42
    @test length(integrator.u_tmp) == 42

    # Test `resize!` for non `VolumeIntegralSubcellLimiting`
    let
        solver = DGSEM(basis, surface_flux)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

        ode = semidiscretize(semi, tspan)
        ode_alg = Trixi.SimpleSSPRK33(stage_callbacks = (;))
        callbacks = CallbackSet(summary_callback)
        integrator = Trixi.init(ode, ode_alg, dt = 11.0, callback = callbacks)

        resize!(integrator, 4711)
        @test length(integrator.u) == 4711
        @test length(integrator.du) == 4711
        @test length(integrator.u_tmp) == 4711
    end
end

@trixi_testset "elixir_euler_sedov.jl with HLLC Flux" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            0.4229948321239887,
                            0.2559038337457483,
                            0.2559038337457484,
                            1.2990046683564136
                        ],
                        linf=[
                            1.4989357969730492,
                            1.325456585141623,
                            1.3254565851416251,
                            6.331283015053501
                        ],
                        surface_flux=flux_hllc,
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl (HLLE)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            0.40853279043747015,
                            0.25356771650524296,
                            0.2535677165052422,
                            1.2984601729572691
                        ],
                        linf=[
                            1.3840909333784284,
                            1.3077772519086124,
                            1.3077772519086157,
                            6.298798630968632
                        ],
                        surface_flux=flux_hlle,
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
                        l2=[
                            0.6321850210104147,
                            0.38691446170269167,
                            0.3868695626809587,
                            1.0657553825683956
                        ],
                        linf=[
                            2.7602280007469666,
                            2.3265993814913672,
                            2.3258078438689673,
                            2.1577683028925416
                        ],
                        tspan=(0.0, 0.3),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_wall_bc_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc_amr.jl"),
                        l2=[
                            0.02026685991647352,
                            0.017467584076280237,
                            0.011378371604813321,
                            0.05138942558296091
                        ],
                        linf=[
                            0.35924402060711524,
                            0.32068389566068806,
                            0.2361141752119986,
                            0.9289840057748628
                        ],
                        tspan=(0.0, 0.15))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_forward_step_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_forward_step_amr.jl"),
                        l2=[
                            0.004191480950848891,
                            0.003781298410569231,
                            0.0013470418422981045,
                            0.03262817609394949
                        ],
                        linf=[
                            2.0581500751947113,
                            2.2051301367971288,
                            3.8502467979250254,
                            17.750333649853616
                        ],
                        tspan=(0.0, 0.0001),
                        rtol=1.0e-7,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_double_mach_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_double_mach_amr.jl"),
                        l2=[
                            0.051359355290192046,
                            0.4266034859911273,
                            0.2438304855475594,
                            4.11487176105527
                        ],
                        linf=[
                            6.902000373057003,
                            53.95714139820832,
                            24.241610279839758,
                            561.0630401858057
                        ],
                        tspan=(0.0, 0.0001),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_supersonic_cylinder.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_supersonic_cylinder.jl"),
                        l2=[
                            0.02676082999794676,
                            0.05110830068968181,
                            0.03205164257040607,
                            0.1965981012724311
                        ],
                        linf=[
                            3.6830683476364476,
                            4.284442685012427,
                            6.857777546171545,
                            31.749285097390576
                        ],
                        tspan=(0.0, 0.001),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_supersonic_cylinder_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_supersonic_cylinder_sc_subcell.jl"),
                        l2=[
                            0.11085870166618325,
                            0.23309905989870722,
                            0.13505351590735631,
                            0.7932047512585592
                        ],
                        linf=[
                            2.9808773737943564,
                            4.209364526217892,
                            6.265341002817672,
                            24.077904874883338
                        ],
                        tspan=(0.0, 0.02),
                        atol=1e-7)
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

@trixi_testset "elixir_euler_NACA6412airfoil_mach2.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_NACA6412airfoil_mach2.jl"),
                        l2=[
                            0.19107654776276498, 0.3545913719444839,
                            0.18492730895077583, 0.817927213517244
                        ],
                        linf=[
                            2.5397624311491946, 2.7075156425517917, 2.200980534211764,
                            9.031153939238115
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        l2=[
                            0.00024871265138964204,
                            0.0003370077102132591,
                            0.0003370077102131964,
                            0.0007231525513793697
                        ],
                        linf=[
                            0.0015813032944647087,
                            0.0020494288423820173,
                            0.0020494288423824614,
                            0.004793821195083758
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[1.0513414461545583e-5, 1.0517900957166411e-6,
                            1.0517900957304043e-6, 1.511816606372376e-6,
                            1.0443997728645063e-6, 7.879639064990798e-7,
                            7.879639065049896e-7, 1.0628631669056271e-6,
                            4.3382328912336153e-7],
                        linf=[4.255466285174592e-5, 1.0029706745823264e-5,
                            1.0029706747467781e-5, 1.2122265939010224e-5,
                            5.4791097160444835e-6, 5.18922042269665e-6,
                            5.189220422141538e-6, 9.552667261422676e-6,
                            1.4237578427628152e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.0322570437144848,
                            0.03598284801272945,
                            0.03562228071357411,
                            0.05288641880143085,
                            0.040752873778199326,
                            0.04207276835260492,
                            0.04171391252403866,
                            0.05289242879893149,
                            0.0016038935411812223
                        ],
                        linf=[
                            0.175984910510666,
                            0.13999726708245439,
                            0.13336032728399658,
                            0.21248359539637798,
                            0.133294808938885,
                            0.17934684696413217,
                            0.1831567822932948,
                            0.21575881133569155,
                            0.01967917976620703
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[0.4551839744017604, 0.8917986079085971, 0.832474072904728,
                            0.0,
                            0.98015167453983, 0.10475978783943254,
                            0.15551175906375883,
                            0.0,
                            2.026208477271868e-5],
                        linf=[10.19496728149964, 18.23726813972206,
                            10.04367783820621,
                            0.0,
                            19.63022306543678, 1.3952679820406384, 1.8716515525771589,
                            0.0,
                            0.0017266639582675424],
                        tspan=(0.0, 0.02))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor_cfl_ramp.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor_cfl_ramp.jl"),
                        l2=[
                            0.45519051169507474,
                            0.8917985468745363,
                            0.8324681609772325,
                            0.0,
                            0.9801426190285389,
                            0.10476233464125001,
                            0.15551270692826116,
                            0.0,
                            2.0201603821472296e-5
                        ],
                        linf=[
                            10.196786739705292,
                            18.267539012179128,
                            10.046104290498878,
                            0.0,
                            19.668302849210974,
                            1.395022093528294,
                            1.8717844606331189,
                            0.0,
                            0.001651262488701531
                        ],
                        tspan=(0.0, 0.02))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_linearizedeuler_gaussian_source.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearizedeuler_gaussian_source.jl"),
                        l2=[
                            0.006047938590548741,
                            0.0040953286019907035,
                            0.004222698522497298,
                            0.006269492499336128
                        ],
                        linf=[
                            0.06386175207349379,
                            0.0378926444850457,
                            0.041759728067967065,
                            0.06430136016259067
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_subsonic_cylinder.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_subsonic_cylinder.jl"),
                        l2=[
                            0.00011914390523852561,
                            0.00010776028621724485,
                            6.139954358305467e-5,
                            0.0003067693731825959
                        ],
                        linf=[
                            0.1653075586200805,
                            0.1868437275544909,
                            0.09772818519679008,
                            0.4311796171737692
                        ], tspan=(0.0, 0.001))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)
    drag = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)
    lift = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)

    @test isapprox(lift, -6.501138753497174e-15, atol = 1e-13)
    @test isapprox(drag, 2.588589856781827, atol = 1e-13)
end

# Forces computation test in an AMR code
@trixi_testset "elixir_euler_NACA0012airfoil_mach085.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_NACA0012airfoil_mach085.jl"),
                        l2=[
                            5.56114097044427e-7, 6.62284247153255e-6,
                            1.0823259724601275e-5, 0.000659804574787503
                        ],
                        linf=[
                            0.002157589754528455, 0.039163189253511164,
                            0.038386804399707625, 2.6685831417913914
                        ],
                        amr_interval=1,
                        base_level=0, med_level=1, max_level=1,
                        tspan=(0.0, 0.0001),
                        adapt_initial_condition=false,
                        adapt_initial_condition_only_refine=false,)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)
    drag = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)
    lift = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)

    @test isapprox(lift, 0.029094009322876882, atol = 1e-13)
    @test isapprox(drag, 0.13579200776643238, atol = 1e-13)
end

@trixi_testset "elixir_euler_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "tree_2d_dgsem"),
                                 "elixir_euler_blast_wave_pure_fv.jl"),
                        l2=[
                            0.39957047631960346,
                            0.21006912294983154,
                            0.21006903549932,
                            0.6280328163981136
                        ],
                        linf=[
                            2.20417889887697,
                            1.5487238480003327,
                            1.5486788679247812,
                            2.4656795949035857
                        ],
                        tspan=(0.0, 0.5),
                        mesh=P4estMesh((64, 64), polydeg = 3,
                                       coordinates_min = (-2.0, -2.0),
                                       coordinates_max = (2.0, 2.0)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_weak_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_amr.jl"),
                        l2=[
                            0.11134260363848127,
                            0.11752357091804219,
                            0.11829112104640764,
                            0.7557891142955036
                        ],
                        linf=[
                            0.5728647031475109,
                            0.8353132977670252,
                            0.8266797080712205,
                            3.9792506230548317
                        ],
                        tspan=(0.0, 0.1),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
    # Check for conservation
    state_integrals = Trixi.integrate(sol.u[2], semi)
    initial_state_integrals = analysis_callback.affect!.initial_state_integrals

    @test isapprox(state_integrals[1], initial_state_integrals[1], atol = 1e-13)
    @test isapprox(state_integrals[2], initial_state_integrals[2], atol = 1e-13)
    @test isapprox(state_integrals[3], initial_state_integrals[3], atol = 1e-13)
    @test isapprox(state_integrals[4], initial_state_integrals[4], atol = 1e-13)
end

@trixi_testset "elixir_euler_SD7003airfoil.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        semi=SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition, solver;
                                                          boundary_conditions = boundary_conditions_hyp),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           output_directory = "out",
                                                           save_analysis = true),
                        l2=[
                            9.316117984455285e-5,
                            4.539266936628966e-5,
                            8.381576796590632e-5,
                            0.00023437941500203496
                        ],
                        linf=[
                            0.31274105032407307,
                            0.2793016762668701,
                            0.22256470161743136,
                            0.7906704256076251
                        ],
                        tspan=(0.0, 5e-3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_density_wave_tracers.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_euler_density_wave_tracers.jl"),
                        l2=[
                            0.0012704690524147188,
                            0.00012704690527390463,
                            0.00025409381047976197,
                            3.17617263147723e-5,
                            0.0527467468452892,
                            0.052788143280791185
                        ],
                        linf=[
                            0.0071511674295154926,
                            0.0007151167435655859,
                            0.0014302334865533006,
                            0.00017877918656949987,
                            0.2247919517756231,
                            0.2779841048041337
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_cylinder_bowshock_mach3.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_euler_cylinder_bowshock_mach3.jl"),
                        tspan=(0.0, 1e-3),
                        l2=[
                            0.03787745781612722,
                            0.03339276348608649,
                            0.05301001151898993,
                            0.2868802674001281
                        ],
                        linf=[
                            2.5347156069842978,
                            2.6657123832452414,
                            3.786891603220761,
                            21.305497055838977
                        ])
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
