
"""
    SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the [`residual_steady_state(du, equations)`](@ref)
falls below the threshold specified by `abstol, reltol`.
"""
mutable struct SteadyStateCallback{RealT<:Real}
  abstol::RealT
  reltol::RealT
end

function SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)
  abstol, reltol = promote(abstol, reltol)
  steady_state_callback = SteadyStateCallback(abstol, reltol)

  DiscreteCallback(steady_state_callback, steady_state_callback,
                   save_positions=(false,false))
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SteadyStateCallback}
  steady_state_callback = cb.affect!
  print(io, "SteadyStateCallback(abstol=", steady_state_callback.abstol, ", ",
                                "reltol=", steady_state_callback.reltol, ")")
end
# TODO: Taal bikeshedding, implement a method with more information and the signature
# function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SteadyStateCallback}
# end


# affect!
(::SteadyStateCallback)(integrator) = terminate!(integrator)


# the condition
function (steady_state_callback::SteadyStateCallback)(u_ode::AbstractVector, t, integrator)
  semi = integrator.p

  u  = wrap_array(u_ode, semi)
  du = wrap_array(get_du(integrator), semi)
  terminate = steady_state_callback(du, u, semi)
  if terminate
    @info "  Steady state tolerance reached" steady_state_callback t
  end
  return terminate
end

function (steady_state_callback::SteadyStateCallback)(du, u, semi::AbstractSemidiscretization)
  steady_state_callback(du, u, mesh_equations_solver_cache(semi)...)
end


include("steady_state_dg2d.jl")
include("steady_state_dg3d.jl")
