function create_cache_parabolic(mesh::DGMultiMesh,
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DGMulti, dg_parabolic, RealT, uEltype)
  # default to taking derivatives of all hyperbolic terms
  # TODO: utilize the parabolic variables in `equations_parabolic` to reduce memory usage in the parabolic cache
  nvars = nvariables(equations_hyperbolic)

  @unpack M, Drst = dg.basis
  weak_differentiation_matrices = map(A -> -M \ (A' * M), Drst)

  # u_transformed stores "transformed" variables for computing the gradient
  @unpack md = mesh
  u_transformed = allocate_nested_array(uEltype, nvars, size(md.x), dg)
  u_grad = ntuple(_ -> similar(u_transformed), ndims(mesh))
  viscous_flux = similar.(u_grad)

  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  scalar_flux_face_values = similar(u_face_values)
  grad_u_face_values = ntuple(_ -> similar(u_face_values), ndims(mesh))

  local_u_values_threaded = [similar(u_transformed, dg.basis.Nq) for _ in 1:Threads.nthreads()]
  local_viscous_flux_threaded = [ntuple(_ -> similar(u_transformed, dg.basis.Nq), ndims(mesh)) for _ in 1:Threads.nthreads()]
  local_flux_face_values_threaded = [similar(scalar_flux_face_values[:, 1]) for _ in 1:Threads.nthreads()]

  # precompute 1 / h for penalty terms
  inv_h = similar(mesh.md.Jf)
  J = dg.basis.Vf * mesh.md.J # interp to face nodes
  for e in eachelement(mesh, dg)
    for i in each_face_node(mesh, dg)
      inv_h[i, e] = mesh.md.Jf[i, e] / J[i, e]
    end
  end

  return (; u_transformed, u_grad, viscous_flux,
            weak_differentiation_matrices, inv_h,
            u_face_values, grad_u_face_values, scalar_flux_face_values,
            local_u_values_threaded, local_viscous_flux_threaded, local_flux_face_values_threaded)
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, extra_args...)
  @threaded for i in eachindex(u)
    u_transformed[i] = u[i]
  end
end

# interpolates from solution coefficients to face quadrature points
function prolong2interfaces!(u_face_values, u, mesh::DGMultiMesh, equations::AbstractEquationsParabolic,
                             surface_integral, dg::DGMulti, cache)
  apply_to_each_field(mul_by!(dg.basis.Vf), u_face_values, u)
end

function calc_gradient_surface_integral(u_grad, u, scalar_flux_face_values,
                                        mesh, equations::AbstractEquationsParabolic,
                                        dg::DGMulti, cache, cache_parabolic)
  @unpack local_flux_face_values_threaded = cache_parabolic
  @threaded for e in eachelement(mesh, dg)
    local_flux_values = local_flux_face_values_threaded[Threads.threadid()]
    for dim in eachdim(mesh)
      for i in eachindex(local_flux_values)
        # compute flux * (nx, ny, nz)
        local_flux_values[i] = scalar_flux_face_values[i, e] * mesh.md.nxyzJ[dim][i, e]
      end
      apply_to_each_field(mul_by_accum!(dg.basis.LIFT), view(u_grad[dim], :, e), local_flux_values)
    end
  end
end

function calc_gradient!(u_grad, u::StructArray, t, mesh::DGMultiMesh,
                        equations::AbstractEquationsParabolic,
                        boundary_conditions, dg::DGMulti, cache, cache_parabolic)

  @unpack weak_differentiation_matrices = cache_parabolic

  for dim in eachindex(u_grad)
    reset_du!(u_grad[dim], dg)
  end

  # compute volume contributions to gradients
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # TODO: assumes mesh is affine
      apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], dxidxhatj),
                          view(u_grad[i], :, e), view(u, :, e))
    end
  end

  @unpack u_face_values = cache_parabolic
  prolong2interfaces!(u_face_values, u, mesh, equations, dg.surface_integral, dg, cache)

  # compute fluxes at interfaces
  @unpack scalar_flux_face_values = cache_parabolic
  @unpack mapM, mapP, Jf = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM = u_face_values[idM]
    uP = u_face_values[idP]
    scalar_flux_face_values[idM] = 0.5 * (uP + uM) # TODO: use strong/weak formulation for curved meshes?
  end

  calc_boundary_flux!(scalar_flux_face_values, nothing, t, Gradient(), boundary_conditions,
                      mesh, equations, dg, cache, cache_parabolic)

  # compute surface contributions
  calc_gradient_surface_integral(u_grad, u, scalar_flux_face_values,
                                 mesh, equations, dg, cache, cache_parabolic)

  for dim in eachdim(mesh)
    invert_jacobian!(u_grad[dim], mesh, equations, dg, cache; scaling=1.0)
  end
end

# do nothing for periodic domains
function calc_boundary_flux!(flux, u, t, operator_type, ::BoundaryConditionPeriodic,
                             mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                             cache, cache_parabolic)
  return nothing
end

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions,
                             mesh, equations, dg::DGMulti, cache, cache_parabolic)

  # peel off first boundary condition
  calc_single_boundary_flux!(flux, u, t, operator_type, first(boundary_conditions), first(keys(boundary_conditions)),
                             mesh, equations, dg, cache, cache_parabolic)

  # recurse on the remainder of the boundary conditions
  calc_boundary_flux!(flux, u, t, operator_type, Base.tail(boundary_conditions),
                      mesh, equations, dg, cache, cache_parabolic)
end

# terminate recursion
calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions::NamedTuple{(),Tuple{}},
                    mesh, equations, dg::DGMulti, cache, cache_parabolic) = nothing

# TODO: DGMulti. Decide if we want to use the input `u_face_values` (currently unused)
function calc_single_boundary_flux!(flux_face_values, u_face_values, t,
                                    operator_type, boundary_condition, boundary_key,
                                    mesh, equations, dg::DGMulti{NDIMS}, cache, cache_parabolic) where {NDIMS}
  rd = dg.basis
  md = mesh.md

  num_pts_per_face = rd.Nfq ÷ rd.Nfaces
  @unpack xyzf, nxyzJ, Jf = md
  for f in mesh.boundary_faces[boundary_key]
    for i in Base.OneTo(num_pts_per_face)

      # reverse engineer element + face node indices (avoids reshaping arrays)
      e = ((f-1) ÷ rd.Nfaces) + 1
      fid = i + ((f-1) % rd.Nfaces) * num_pts_per_face

      face_normal = SVector{NDIMS}(getindex.(nxyzJ, fid, e)) / Jf[fid,e]
      face_coordinates = SVector{NDIMS}(getindex.(xyzf, fid, e))

      # for both the gradient and the divergence, the boundary flux is scalar valued.
      # for the gradient, it is the solution; for divergence, it is the normal flux.
      u_boundary = boundary_condition(flux_face_values[fid,e],
                                      face_normal, face_coordinates, t,
                                      operator_type, equations)
      flux_face_values[fid,e] = u_boundary
    end
  end
  return nothing
end

function calc_viscous_fluxes!(viscous_flux, u, u_grad, mesh::DGMultiMesh,
                              equations::AbstractEquationsParabolic,
                              dg::DGMulti, cache, cache_parabolic)

  for dim in eachdim(mesh)
    reset_du!(viscous_flux[dim], dg)
  end

  @unpack local_viscous_flux_threaded, local_u_values_threaded = cache_parabolic

  @threaded for e in eachelement(mesh, dg)

    # reset local storage for each element
    local_viscous_flux = local_viscous_flux_threaded[Threads.threadid()]
    local_u_values = local_u_values_threaded[Threads.threadid()]
    fill!(local_u_values, zero(eltype(local_u_values)))
    for dim in eachdim(mesh)
      fill!(local_viscous_flux[dim], zero(eltype(local_viscous_flux[dim])))
    end

    # interpolate u and gradient to quadrature points, store in `local_viscous_flux`
    apply_to_each_field(mul_by!(dg.basis.Vq), local_u_values, view(u, :, e)) # TODO: can we avoid this when we don't need it?
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Vq), local_viscous_flux[dim], view(u_grad[dim], :, e))
    end

    # compute viscous flux at quad points
    for i in eachindex(local_u_values)
      u_i = local_u_values[i]
      u_grad_i = getindex.(local_viscous_flux, i) # TODO: check if this allocates. Shouldn't for tuples or SVector...
      viscous_flux_i = flux(u_i, u_grad_i, equations)
      setindex!.(local_viscous_flux, viscous_flux_i, i)
    end

    # project back to the DG approximation space
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Pq), view(viscous_flux[dim], :, e), local_viscous_flux[dim])
    end
  end
end

# no penalization for a BR1 parabolic solver
function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               dg_parabolic::ViscousFormulationBassiRebay1, cache, cache_parabolic)
  return nothing
end

function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               dg_parabolic, cache, cache_parabolic)
  # compute fluxes at interfaces
  @unpack scalar_flux_face_values, inv_h = cache_parabolic
  @unpack mapM, mapP = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM, uP = u_face_values[idM], u_face_values[idP]
    inv_h_face = inv_h[face_node_index]
    scalar_flux_face_values[idM] = scalar_flux_face_values[idM] + penalty(uP, uM, inv_h_face, equations, dg_parabolic)
  end
  return nothing
end


function calc_divergence!(du, u::StructArray, t, viscous_flux, mesh::DGMultiMesh,
                          equations::AbstractEquationsParabolic,
                          boundary_conditions, dg::DGMulti, dg_parabolic, cache, cache_parabolic)

  @unpack weak_differentiation_matrices = cache_parabolic

  reset_du!(du, dg)

  # compute volume contributions to divergence
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
      apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], dxidxhatj),
                                view(du, :, e), view(viscous_flux[i], :, e))
    end
  end

  # interpolates from solution coefficients to face quadrature points
  viscous_flux_face_values = cache_parabolic.grad_u_face_values # reuse storage
  for dim in eachdim(mesh)
    prolong2interfaces!(viscous_flux_face_values[dim], viscous_flux[dim], mesh, equations,
                        dg.surface_integral, dg, cache)
  end

  # compute fluxes at interfaces
  @unpack scalar_flux_face_values = cache_parabolic
  @unpack mapM, mapP, nxyzJ = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg, cache, cache_parabolic)
    idM, idP = mapM[face_node_index], mapP[face_node_index]

    # compute f(u, ∇u) ⋅ n
    flux_face_value = zero(eltype(scalar_flux_face_values))
    for dim in eachdim(mesh)
      uM = viscous_flux_face_values[dim][idM]
      uP = viscous_flux_face_values[dim][idP]
      # TODO: use strong/weak formulation?
      flux_face_value = flux_face_value + 0.5 * (uP + uM) * nxyzJ[dim][face_node_index]
    end
    scalar_flux_face_values[idM] = flux_face_value
  end

  # TODO: decide what to pass in
  calc_boundary_flux!(scalar_flux_face_values, nothing, t, Divergence(),
                      boundary_conditions, mesh, equations, dg, cache, cache_parabolic)

  calc_viscous_penalty!(scalar_flux_face_values, cache_parabolic.u_face_values, t,
                        boundary_conditions, mesh, equations, dg, dg_parabolic,
                        cache, cache_parabolic)

  # surface contributions
  apply_to_each_field(mul_by_accum!(dg.basis.LIFT), du, scalar_flux_face_values)

  invert_jacobian!(du, mesh, equations, dg, cache; scaling=1.0)
end

# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(u)
# boundary conditions will be applied to both grad(u) and div(u).
function rhs_parabolic!(du, u, t, mesh::DGMultiMesh, equations_parabolic::AbstractEquationsParabolic,
                        initial_condition, boundary_conditions, source_terms,
                        dg::DGMulti, dg_parabolic, cache, cache_parabolic)

  reset_du!(du, dg)

  @unpack u_transformed, u_grad, viscous_flux = cache_parabolic
  transform_variables!(u_transformed, u, equations_parabolic)

  calc_gradient!(u_grad, u_transformed, t, mesh, equations_parabolic,
                 boundary_conditions, dg, cache, cache_parabolic)

  calc_viscous_fluxes!(viscous_flux, u_transformed, u_grad,
                       mesh, equations_parabolic, dg, cache, cache_parabolic)

  calc_divergence!(du, u_transformed, t, viscous_flux, mesh, equations_parabolic,
                   boundary_conditions, dg, dg_parabolic, cache, cache_parabolic)

  return nothing

end