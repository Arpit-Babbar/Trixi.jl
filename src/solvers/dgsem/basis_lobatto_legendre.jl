# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    LobattoLegendreBasis([RealT = Float64,] polydeg::Integer)

Create a nodal Lobatto-Legendre basis for polynomials of degree `polydeg`.

For the special case `polydeg=0` the DG method reduces to a finite volume method.
Therefore, this function sets the center point of the cell as single node.
This exceptional case is currently only supported for TreeMesh!
"""
struct LobattoLegendreBasis{RealT <: Real, NNODES,
                            VectorT <: AbstractVector{RealT},
                            InverseVandermondeLegendre <: AbstractMatrix{RealT},
                            BoundaryMatrix <: AbstractMatrix{RealT},
                            DerivativeMatrix <: AbstractMatrix{RealT}} <:
       AbstractBasisSBP{RealT}
    nodes::VectorT
    weights::VectorT
    inverse_weights::VectorT

    inverse_vandermonde_legendre::InverseVandermondeLegendre
    boundary_interpolation::BoundaryMatrix # lhat

    derivative_matrix::DerivativeMatrix # strong form derivative matrix
    derivative_split::DerivativeMatrix # strong form derivative matrix minus boundary terms
    derivative_split_transpose::DerivativeMatrix # transpose of `derivative_split`
    derivative_dhat::DerivativeMatrix # weak form matrix "dhat",
    # negative adjoint wrt the SBP dot product
end

function Adapt.adapt_structure(to, basis::LobattoLegendreBasis)
    inverse_vandermonde_legendre = adapt(to, basis.inverse_vandermonde_legendre)
    RealT = eltype(inverse_vandermonde_legendre)

    nodes = SVector{<:Any, RealT}(basis.nodes)
    weights = SVector{<:Any, RealT}(basis.weights)
    inverse_weights = SVector{<:Any, RealT}(basis.inverse_weights)
    boundary_interpolation = adapt(to, basis.boundary_interpolation)
    derivative_matrix = adapt(to, basis.derivative_matrix)
    derivative_split = adapt(to, basis.derivative_split)
    derivative_split_transpose = adapt(to, basis.derivative_split_transpose)
    derivative_dhat = adapt(to, basis.derivative_dhat)
    return LobattoLegendreBasis{RealT, nnodes(basis), typeof(nodes),
                                typeof(inverse_vandermonde_legendre),
                                typeof(boundary_interpolation),
                                typeof(derivative_matrix)}(nodes,
                                                           weights,
                                                           inverse_weights,
                                                           inverse_vandermonde_legendre,
                                                           boundary_interpolation,
                                                           derivative_matrix,
                                                           derivative_split,
                                                           derivative_split_transpose,
                                                           derivative_dhat)
end

function LobattoLegendreBasis(RealT, polydeg::Integer)
    nnodes_ = polydeg + 1

    nodes_, weights_ = gauss_lobatto_nodes_weights(nnodes_, RealT)
    inverse_weights_ = inv.(weights_)

    _, inverse_vandermonde_legendre = vandermonde_legendre(nodes_, RealT)

    boundary_interpolation = zeros(RealT, nnodes_, 2)
    boundary_interpolation[:, 1] = calc_lhat(-one(RealT), nodes_, weights_)
    boundary_interpolation[:, 2] = calc_lhat(one(RealT), nodes_, weights_)

    derivative_matrix = polynomial_derivative_matrix(nodes_)
    derivative_split = calc_dsplit(nodes_, weights_)
    derivative_split_transpose = Matrix(derivative_split')
    derivative_dhat = calc_dhat(nodes_, weights_)

    # Type conversions to enable possible optimizations of runtime performance 
    # and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)
    inverse_weights = SVector{nnodes_, RealT}(inverse_weights_)

    # We keep the matrices above stored using the standard `Matrix` type 
    # since this is usually as fast as `SMatrix`
    # (when using `let` in the volume integral/`@threaded`)
    # and reduces latency

    return LobattoLegendreBasis{RealT, nnodes_, typeof(nodes),
                                typeof(inverse_vandermonde_legendre),
                                typeof(boundary_interpolation),
                                typeof(derivative_matrix)}(nodes, weights,
                                                           inverse_weights,
                                                           inverse_vandermonde_legendre,
                                                           boundary_interpolation,
                                                           derivative_matrix,
                                                           derivative_split,
                                                           derivative_split_transpose,
                                                           derivative_dhat)
end
LobattoLegendreBasis(polydeg::Integer) = LobattoLegendreBasis(Float64, polydeg)

function Base.show(io::IO, basis::LobattoLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "LobattoLegendreBasis{", real(basis), "}(polydeg=", polydeg(basis), ")")
end
function Base.show(io::IO, ::MIME"text/plain", basis::LobattoLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "LobattoLegendreBasis{", real(basis), "} with polynomials of degree ",
          polydeg(basis))
end

function Base.:(==)(b1::LobattoLegendreBasis, b2::LobattoLegendreBasis)
    if typeof(b1) != typeof(b2)
        return false
    end

    for field in fieldnames(typeof(b1))
        if getfield(b1, field) != getfield(b2, field)
            return false
        end
    end

    return true
end

@inline Base.real(basis::LobattoLegendreBasis{RealT}) where {RealT} = RealT

@inline function nnodes(basis::LobattoLegendreBasis{RealT, NNODES}) where {RealT, NNODES
                                                                           }
    NNODES
end

"""
    eachnode(basis::LobattoLegendreBasis)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `basis`. 
In particular, not the nodes themselves are returned.
"""
@inline eachnode(basis::LobattoLegendreBasis) = Base.OneTo(nnodes(basis))

@inline polydeg(basis::LobattoLegendreBasis) = nnodes(basis) - 1

@inline get_nodes(basis::LobattoLegendreBasis) = basis.nodes

"""
    integrate(f, u, basis::LobattoLegendreBasis)

Map the function `f` to the coefficients `u` and integrate with respect to the
quadrature rule given by `basis`.
"""
function integrate(f, u, basis::LobattoLegendreBasis)
    @unpack weights = basis

    res = zero(f(first(u)))
    for i in eachindex(u, weights)
        res += f(u[i]) * weights[i]
    end
    return res
end

# Return the first/last weight of the quadrature associated with `basis`.
# Since the mass matrix for nodal Lobatto-Legendre bases is diagonal,
# these weights are the only coefficients necessary for the scaling of
# surface terms/integrals in DGSEM.
left_boundary_weight(basis::LobattoLegendreBasis) = first(basis.weights)
right_boundary_weight(basis::LobattoLegendreBasis) = last(basis.weights)

struct LobattoLegendreMortarL2{RealT <: Real, NNODES,
                               ForwardMatrix <: AbstractMatrix{RealT},
                               ReverseMatrix <: AbstractMatrix{RealT}} <:
       AbstractMortarL2{RealT}
    forward_upper::ForwardMatrix
    forward_lower::ForwardMatrix
    reverse_upper::ReverseMatrix
    reverse_lower::ReverseMatrix
end

function Adapt.adapt_structure(to, mortar::LobattoLegendreMortarL2)
    forward_upper = adapt(to, mortar.forward_upper)
    forward_lower = adapt(to, mortar.forward_lower)
    reverse_upper = adapt(to, mortar.reverse_upper)
    reverse_lower = adapt(to, mortar.reverse_lower)
    return LobattoLegendreMortarL2{eltype(forward_upper), nnodes(mortar),
                                   typeof(forward_upper),
                                   typeof(reverse_upper)}(forward_upper, forward_lower,
                                                          reverse_upper, reverse_lower)
end

function MortarL2(basis::LobattoLegendreBasis)
    RealT = real(basis)
    nnodes_ = nnodes(basis)

    forward_upper = calc_forward_upper(nnodes_, RealT)
    forward_lower = calc_forward_lower(nnodes_, RealT)
    reverse_upper = calc_reverse_upper(nnodes_, Val(:gauss), RealT)
    reverse_lower = calc_reverse_lower(nnodes_, Val(:gauss), RealT)

    # We keep the matrices above stored using the standard `Matrix` type 
    # since this is usually as fast as `SMatrix`
    # (when using `let` in the volume integral/`@threaded`)
    # and reduces latency

    # TODO: Taal performance
    #       Check the performance of different implementations of `mortar_fluxes_to_elements!`
    #       with different types of the reverse matrices and different types of
    #       `fstar_upper_threaded` etc. used in the cache.
    #       Check whether `@turbo` with `eachnode` in `multiply_dimensionwise!` can be faster than
    #       `@tullio` when the matrix sizes are not necessarily static.
    # reverse_upper = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(reverse_upper_)
    # reverse_lower = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(reverse_lower_)

    LobattoLegendreMortarL2{RealT, nnodes_, typeof(forward_upper),
                            typeof(reverse_upper)}(forward_upper, forward_lower,
                                                   reverse_upper, reverse_lower)
end

function Base.show(io::IO, mortar::LobattoLegendreMortarL2)
    @nospecialize mortar # reduce precompilation time

    print(io, "LobattoLegendreMortarL2{", real(mortar), "}(polydeg=", polydeg(mortar),
          ")")
end
function Base.show(io::IO, ::MIME"text/plain", mortar::LobattoLegendreMortarL2)
    @nospecialize mortar # reduce precompilation time

    print(io, "LobattoLegendreMortarL2{", real(mortar), "} with polynomials of degree ",
          polydeg(mortar))
end

@inline Base.real(mortar::LobattoLegendreMortarL2{RealT}) where {RealT} = RealT

@inline function nnodes(mortar::LobattoLegendreMortarL2{RealT, NNODES}) where {RealT,
                                                                               NNODES}
    NNODES
end

@inline polydeg(mortar::LobattoLegendreMortarL2) = nnodes(mortar) - 1

# TODO: We can create EC mortars along the lines of the following implementation.
# abstract type AbstractMortarEC{RealT} <: AbstractMortar{RealT} end

# struct LobattoLegendreMortarEC{RealT<:Real, NNODES, MortarMatrix<:AbstractMatrix{RealT}, SurfaceFlux} <: AbstractMortarEC{RealT}
#   forward_upper::MortarMatrix
#   forward_lower::MortarMatrix
#   reverse_upper::MortarMatrix
#   reverse_lower::MortarMatrix
#   surface_flux::SurfaceFlux
# end

# function MortarEC(basis::LobattoLegendreBasis{RealT}, surface_flux)
#   forward_upper   = calc_forward_upper(n_nodes, RealT)
#   forward_lower   = calc_forward_lower(n_nodes, RealT)
#   l2reverse_upper = calc_reverse_upper(n_nodes, Val(:gauss_lobatto), RealT)
#   l2reverse_lower = calc_reverse_lower(n_nodes, Val(:gauss_lobatto), RealT)

#   # type conversions to make use of StaticArrays etc.
#   nnodes_ = nnodes(basis)
#   forward_upper   = SMatrix{nnodes_, nnodes_}(forward_upper)
#   forward_lower   = SMatrix{nnodes_, nnodes_}(forward_lower)
#   l2reverse_upper = SMatrix{nnodes_, nnodes_}(l2reverse_upper)
#   l2reverse_lower = SMatrix{nnodes_, nnodes_}(l2reverse_lower)

#   LobattoLegendreMortarEC{RealT, nnodes_, typeof(forward_upper), typeof(surface_flux)}(
#     forward_upper, forward_lower,
#     l2reverse_upper, l2reverse_lower,
#     surface_flux
#   )
# end

# @inline nnodes(mortar::LobattoLegendreMortarEC{RealT, NNODES}) = NNODES

struct LobattoLegendreAnalyzer{RealT <: Real, NNODES,
                               VectorT <: AbstractVector{RealT},
                               Vandermonde <: AbstractMatrix{RealT}} <:
       SolutionAnalyzer{RealT}
    nodes::VectorT
    weights::VectorT
    vandermonde::Vandermonde
end

function SolutionAnalyzer(basis::LobattoLegendreBasis;
                          analysis_polydeg = 2 * polydeg(basis))
    RealT = real(basis)
    nnodes_ = analysis_polydeg + 1

    nodes_, weights_ = gauss_lobatto_nodes_weights(nnodes_, RealT)
    vandermonde = polynomial_interpolation_matrix(get_nodes(basis), nodes_)

    # Type conversions to enable possible optimizations of runtime performance 
    # and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)

    return LobattoLegendreAnalyzer{RealT, nnodes_, typeof(nodes), typeof(vandermonde)}(nodes,
                                                                                       weights,
                                                                                       vandermonde)
end

function Base.show(io::IO, analyzer::LobattoLegendreAnalyzer)
    @nospecialize analyzer # reduce precompilation time

    print(io, "LobattoLegendreAnalyzer{", real(analyzer), "}(polydeg=",
          polydeg(analyzer), ")")
end
function Base.show(io::IO, ::MIME"text/plain", analyzer::LobattoLegendreAnalyzer)
    @nospecialize analyzer # reduce precompilation time

    print(io, "LobattoLegendreAnalyzer{", real(analyzer),
          "} with polynomials of degree ", polydeg(analyzer))
end

@inline Base.real(analyzer::LobattoLegendreAnalyzer{RealT}) where {RealT} = RealT

@inline function nnodes(analyzer::LobattoLegendreAnalyzer{RealT, NNODES}) where {RealT,
                                                                                 NNODES}
    NNODES
end
"""
    eachnode(analyzer::LobattoLegendreAnalyzer)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `analyzer`. 
In particular, not the nodes themselves are returned.
"""
@inline eachnode(analyzer::LobattoLegendreAnalyzer) = Base.OneTo(nnodes(analyzer))

@inline polydeg(analyzer::LobattoLegendreAnalyzer) = nnodes(analyzer) - 1

struct LobattoLegendreAdaptorL2{RealT <: Real, NNODES,
                                ForwardMatrix <: AbstractMatrix{RealT},
                                ReverseMatrix <: AbstractMatrix{RealT}} <:
       AdaptorL2{RealT}
    forward_upper::ForwardMatrix
    forward_lower::ForwardMatrix
    reverse_upper::ReverseMatrix
    reverse_lower::ReverseMatrix
end

function AdaptorL2(basis::LobattoLegendreBasis{RealT}) where {RealT}
    nnodes_ = nnodes(basis)

    forward_upper_ = calc_forward_upper(nnodes_, RealT)
    forward_lower_ = calc_forward_lower(nnodes_, RealT)
    reverse_upper_ = calc_reverse_upper(nnodes_, Val(:gauss), RealT)
    reverse_lower_ = calc_reverse_lower(nnodes_, Val(:gauss), RealT)

    # type conversions to get the requested real type and enable possible
    # optimizations of runtime performance and latency

    # TODO: Taal performance
    #       Check the performance of different implementations of
    #       `refine_elements!` (forward) and `coarsen_elements!` (reverse)
    #       with different types of the matrices.
    #       Check whether `@turbo` with `eachnode` in `multiply_dimensionwise!`
    #       can be faster than `@tullio` when the matrix sizes are not necessarily
    #       static.
    forward_upper = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(forward_upper_)
    forward_lower = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(forward_lower_)
    # forward_upper = Matrix{RealT}(forward_upper_)
    # forward_lower = Matrix{RealT}(forward_lower_)

    reverse_upper = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(reverse_upper_)
    reverse_lower = SMatrix{nnodes_, nnodes_, RealT, nnodes_^2}(reverse_lower_)
    # reverse_upper = Matrix{RealT}(reverse_upper_)
    # reverse_lower = Matrix{RealT}(reverse_lower_)

    LobattoLegendreAdaptorL2{RealT, nnodes_, typeof(forward_upper),
                             typeof(reverse_upper)}(forward_upper, forward_lower,
                                                    reverse_upper, reverse_lower)
end

function Base.show(io::IO, adaptor::LobattoLegendreAdaptorL2)
    @nospecialize adaptor # reduce precompilation time

    print(io, "LobattoLegendreAdaptorL2{", real(adaptor), "}(polydeg=",
          polydeg(adaptor), ")")
end
function Base.show(io::IO, ::MIME"text/plain", adaptor::LobattoLegendreAdaptorL2)
    @nospecialize adaptor # reduce precompilation time

    print(io, "LobattoLegendreAdaptorL2{", real(adaptor),
          "} with polynomials of degree ", polydeg(adaptor))
end

@inline Base.real(adaptor::LobattoLegendreAdaptorL2{RealT}) where {RealT} = RealT

@inline function nnodes(adaptor::LobattoLegendreAdaptorL2{RealT, NNODES}) where {RealT,
                                                                                 NNODES}
    NNODES
end

@inline polydeg(adaptor::LobattoLegendreAdaptorL2) = nnodes(adaptor) - 1

###############################################################################
# Polynomial derivative and interpolation functions

# TODO: Taal refactor, allow other RealT below and adapt constructors above accordingly

# Calculate the Dhat matrix
function calc_dhat(nodes, weights)
    n_nodes = length(nodes)
    dhat = Matrix(polynomial_derivative_matrix(nodes)')

    for n in 1:n_nodes, j in 1:n_nodes
        dhat[j, n] *= -weights[n] / weights[j]
    end

    return dhat
end

# Calculate the Dsplit matrix for split-form differentiation: dplit = 2D - M⁻¹B
function calc_dsplit(nodes, weights)
    # Start with 2 x the normal D matrix
    dsplit = 2 .* polynomial_derivative_matrix(nodes)

    # Modify to account for
    dsplit[1, 1] += 1 / weights[1]
    dsplit[end, end] -= 1 / weights[end]

    return dsplit
end

# Calculate the polynomial derivative matrix D.
# This implements algorithm 37 "PolynomialDerivativeMatrix" from Kopriva's book.
function polynomial_derivative_matrix(nodes)
    n_nodes = length(nodes)
    d = zeros(eltype(nodes), n_nodes, n_nodes)
    wbary = barycentric_weights(nodes)

    for i in 1:n_nodes, j in 1:n_nodes
        if j != i
            d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
            d[i, i] -= d[i, j]
        end
    end

    return d
end

# Calculate and interpolation matrix (Vandermonde matrix) between two given sets of nodes
# See algorithm 32 "PolynomialInterpolationMatrix" from Kopriva's book.
function polynomial_interpolation_matrix(nodes_in, nodes_out,
                                         baryweights_in = barycentric_weights(nodes_in))
    n_nodes_in = length(nodes_in)
    n_nodes_out = length(nodes_out)
    vandermonde = Matrix{promote_type(eltype(nodes_in), eltype(nodes_out))}(undef,
                                                                            n_nodes_out,
                                                                            n_nodes_in)
    polynomial_interpolation_matrix!(vandermonde, nodes_in, nodes_out, baryweights_in)

    return vandermonde
end

# This implements algorithm 32 "PolynomialInterpolationMatrix" from Kopriva's book.
function polynomial_interpolation_matrix!(vandermonde,
                                          nodes_in, nodes_out,
                                          baryweights_in)
    fill!(vandermonde, zero(eltype(vandermonde)))

    for k in eachindex(nodes_out)
        match = false
        for j in eachindex(nodes_in)
            if isapprox(nodes_out[k], nodes_in[j])
                match = true
                vandermonde[k, j] = 1
            end
        end

        if match == false
            s = zero(eltype(vandermonde))
            for j in eachindex(nodes_in)
                t = baryweights_in[j] / (nodes_out[k] - nodes_in[j])
                vandermonde[k, j] = t
                s += t
            end
            for j in eachindex(nodes_in)
                vandermonde[k, j] = vandermonde[k, j] / s
            end
        end
    end

    return vandermonde
end

"""
    barycentric_weights(nodes)

Calculate the barycentric weights for a given node distribution, i.e.,
```math
w_j = \\frac{1}{ \\prod_{k \\neq j} \\left( x_j - x_k \\right ) }
```

For details, see (especially Section 3)
- Jean-Paul Berrut and Lloyd N. Trefethen (2004).
  Barycentric Lagrange Interpolation.
  [DOI:10.1137/S0036144502417715](https://doi.org/10.1137/S0036144502417715)
"""
function barycentric_weights(nodes)
    n_nodes = length(nodes)
    weights = ones(eltype(nodes), n_nodes)

    for j in 2:n_nodes, k in 1:(j - 1)
        weights[k] *= nodes[k] - nodes[j]
        weights[j] *= nodes[j] - nodes[k]
    end

    for j in 1:n_nodes
        weights[j] = 1 / weights[j]
    end

    return weights
end

# Calculate Lhat.
function calc_lhat(x, nodes, weights)
    n_nodes = length(nodes)
    wbary = barycentric_weights(nodes)

    lhat = lagrange_interpolating_polynomials(x, nodes, wbary)

    for i in 1:n_nodes
        lhat[i] /= weights[i]
    end

    return lhat
end

""" 
    lagrange_interpolating_polynomials(x, nodes, wbary)

Calculate Lagrange polynomials for a given node distribution with
associated barycentric weights `wbary` at a given point `x` on the 
reference interval ``[-1, 1]``.

This returns all ``l_j(x)``, i.e., the Lagrange polynomials for each node ``x_j``.
Thus, to obtain the interpolating polynomial ``p(x)`` at ``x``, one has to 
multiply the Lagrange polynomials with the nodal values ``u_j`` and sum them up:
``p(x) = \\sum_{j=1}^{n} u_j l_j(x)``.

For details, see e.g. Section 2 of 
- Jean-Paul Berrut and Lloyd N. Trefethen (2004).
  Barycentric Lagrange Interpolation.
  [DOI:10.1137/S0036144502417715](https://doi.org/10.1137/S0036144502417715)
"""
function lagrange_interpolating_polynomials(x, nodes, wbary)
    n_nodes = length(nodes)
    polynomials = zeros(eltype(nodes), n_nodes)

    for i in 1:n_nodes
        # Avoid division by zero when `x` is close to node by using 
        # the Kronecker-delta property at nodes
        # of the Lagrange interpolation polynomials.
        if isapprox(x, nodes[i], rtol = eps(x))
            polynomials[i] = 1
            return polynomials
        end
    end

    for i in 1:n_nodes
        polynomials[i] = wbary[i] / (x - nodes[i])
    end
    total = sum(polynomials)

    for i in 1:n_nodes
        polynomials[i] /= total
    end

    return polynomials
end

"""
    gauss_lobatto_nodes_weights(n_nodes::Integer, RealT = Float64)

Computes nodes ``x_j`` and weights ``w_j`` for the (Legendre-)Gauss-Lobatto quadrature.
This implements algorithm 25 "GaussLobattoNodesAndWeights" from the book

- David A. Kopriva, (2009). 
  Implementing spectral methods for partial differential equations:
  Algorithms for scientists and engineers. 
  [DOI:10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
function gauss_lobatto_nodes_weights(n_nodes::Integer, RealT = Float64)
    # From FLUXO (but really from blue book by Kopriva)
    n_iterations = 20
    tolerance = 2 * eps(RealT) # Relative tolerance for Newton iteration

    # Initialize output
    nodes = zeros(RealT, n_nodes)
    weights = zeros(RealT, n_nodes)

    # Special case for polynomial degree zero (first order finite volume)
    if n_nodes == 1
        nodes[1] = 0
        weights[1] = 2
        return nodes, weights
    end

    # Get polynomial degree for convenience
    N = n_nodes - 1

    # Calculate values at boundary
    nodes[1] = -1
    nodes[end] = 1
    weights[1] = RealT(2) / (N * (N + 1))
    weights[end] = weights[1]

    # Calculate interior values
    if N > 1
        cont1 = convert(RealT, pi) / N
        cont2 = 3 / (8 * N * convert(RealT, pi))

        # Use symmetry -> only left side is computed
        for i in 1:(div(N + 1, 2) - 1)
            # Calculate node
            # Initial guess for Newton method
            nodes[i + 1] = -cos(cont1 * (i + 0.25) - cont2 / (i + 0.25))

            # Newton iteration to find root of Legendre polynomial (= integration node)
            for k in 0:n_iterations
                q, qder, _ = calc_q_and_l(N, nodes[i + 1])
                dx = -q / qder
                nodes[i + 1] += dx
                if abs(dx) < tolerance * abs(nodes[i + 1])
                    break
                end

                if k == n_iterations
                    @warn "`gauss_lobatto_nodes_weights` Newton iteration did not converge"
                end
            end

            # Calculate weight
            _, _, L = calc_q_and_l(N, nodes[i + 1])
            weights[i + 1] = weights[1] / L^2

            # Set nodes and weights according to symmetry properties
            nodes[N + 1 - i] = -nodes[i + 1]
            weights[N + 1 - i] = weights[i + 1]
        end
    end

    # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
    if n_nodes % 2 == 1
        _, _, L = calc_q_and_l(N, zero(RealT))
        nodes[div(N, 2) + 1] = 0
        weights[div(N, 2) + 1] = weights[1] / L^2
    end

    return nodes, weights
end

# From FLUXO (but really from blue book by Kopriva, algorithm 24)
function calc_q_and_l(N::Integer, x::Real)
    RealT = typeof(x)

    L_Nm2 = one(RealT)
    L_Nm1 = x
    Lder_Nm2 = zero(RealT)
    Lder_Nm1 = one(RealT)

    local L
    for i in 2:N
        L = ((2 * i - 1) * x * L_Nm1 - (i - 1) * L_Nm2) / i
        Lder = Lder_Nm2 + (2 * i - 1) * L_Nm1
        L_Nm2 = L_Nm1
        L_Nm1 = L
        Lder_Nm2 = Lder_Nm1
        Lder_Nm1 = Lder
    end

    q = (2 * N + 1) / (N + 1) * (x * L - L_Nm2)
    qder = (2 * N + 1) * L

    return q, qder, L
end

"""
    gauss_nodes_weights(n_nodes::Integer, RealT = Float64)

Computes nodes ``x_j`` and weights ``w_j`` for the Gauss-Legendre quadrature.
This implements algorithm 23 "LegendreGaussNodesAndWeights" from the book

- David A. Kopriva, (2009). 
  Implementing spectral methods for partial differential equations:
  Algorithms for scientists and engineers. 
  [DOI:10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
function gauss_nodes_weights(n_nodes::Integer, RealT = Float64)
    n_iterations = 20
    tolerance = 2 * eps(RealT) # Relative tolerance for Newton iteration

    # Initialize output
    nodes = ones(RealT, n_nodes)
    weights = zeros(RealT, n_nodes)

    # Get polynomial degree for convenience
    N = n_nodes - 1
    if N == 0
        nodes .= 0
        weights .= 2
        return nodes, weights
    elseif N == 1
        nodes[1] = -sqrt(one(RealT) / 3)
        nodes[end] = -nodes[1]
        weights .= 1
        return nodes, weights
    else # N > 1
        # Use symmetry property of the roots of the Legendre polynomials
        for i in 0:(div(N + 1, 2) - 1)
            # Starting guess for Newton method
            nodes[i + 1] = -cospi(one(RealT) / (2 * N + 2) * (2 * i + 1))

            # Newton iteration to find root of Legendre polynomial (= integration node)
            for k in 0:n_iterations
                poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i + 1])
                dx = -poly / deriv
                nodes[i + 1] += dx
                if abs(dx) < tolerance * abs(nodes[i + 1])
                    break
                end

                if k == n_iterations
                    @warn "`gauss_nodes_weights` Newton iteration did not converge"
                end
            end

            # Calculate weight
            poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i + 1])
            weights[i + 1] = (2 * N + 3) / ((1 - nodes[i + 1]^2) * deriv^2)

            # Set nodes and weights according to symmetry properties
            nodes[N + 1 - i] = -nodes[i + 1]
            weights[N + 1 - i] = weights[i + 1]
        end

        # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
        if n_nodes % 2 == 1
            poly, deriv = legendre_polynomial_and_derivative(N + 1, zero(RealT))
            nodes[div(N, 2) + 1] = 0
            weights[div(N, 2) + 1] = (2 * N + 3) / deriv^2
        end

        return nodes, weights
    end
end

"""
    legendre_polynomial_and_derivative(N::Int, x::Real)

Computes the Legendre polynomial of degree `N` and its derivative at `x`.
This implements algorithm 22 "LegendrePolynomialAndDerivative" from the book

- David A. Kopriva, (2009). 
  Implementing spectral methods for partial differential equations:
  Algorithms for scientists and engineers. 
  [DOI:10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
function legendre_polynomial_and_derivative(N::Int, x::Real)
    RealT = typeof(x)
    if N == 0
        poly = one(RealT)
        deriv = zero(RealT)
    elseif N == 1
        poly = x
        deriv = one(RealT)
    else
        poly_Nm2 = one(RealT)
        poly_Nm1 = copy(x)
        deriv_Nm2 = zero(RealT)
        deriv_Nm1 = one(RealT)

        poly = zero(RealT)
        deriv = zero(RealT)
        for i in 2:N
            poly = ((2 * i - 1) * x * poly_Nm1 - (i - 1) * poly_Nm2) / i
            deriv = deriv_Nm2 + (2 * i - 1) * poly_Nm1
            poly_Nm2 = poly_Nm1
            poly_Nm1 = poly
            deriv_Nm2 = deriv_Nm1
            deriv_Nm1 = deriv
        end
    end

    # Normalize
    poly = poly * sqrt(N + 0.5)
    deriv = deriv * sqrt(N + 0.5)

    return poly, deriv
end

# Calculate Legendre vandermonde matrix and its inverse
function vandermonde_legendre(nodes, N::Integer, RealT = Float64)
    n_nodes = length(nodes)
    n_modes = N + 1
    vandermonde = zeros(RealT, n_nodes, n_modes)

    for i in 1:n_nodes
        for m in 1:n_modes
            vandermonde[i, m], _ = legendre_polynomial_and_derivative(m - 1, nodes[i])
        end
    end
    # for very high polynomial degree, this is not well conditioned
    inverse_vandermonde = inv(vandermonde)
    return vandermonde, inverse_vandermonde
end
vandermonde_legendre(nodes, RealT = Float64) = vandermonde_legendre(nodes,
                                                                    length(nodes) - 1,
                                                                    RealT)
end # @muladd
