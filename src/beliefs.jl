struct LBPWNodeBelief{S,A,O,P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    LBPWNodeBelief{S,A,O,P}(m,a,o,d) where {S,A,O,P} = new(m,a,o,d)
    function LBPWNodeBelief{S, A, O, P}(m::P, s::S, a::A, sp::S, o::O, r) where {S, A, O, P}
        cv = CategoricalVector{Tuple{S,Float64}}((sp, convert(Float64, r)),
                                                 obs_weight(m, s, a, sp, o))
        new(m, a, o, cv)
    end
end

function LBPWNodeBelief(model::POMDP{S,A,O}, s::S, a::A, sp::S, o::O, r) where {S,A,O}
    LBPWNodeBelief{S,A,O,typeof(model)}(model, s, a, sp, o, r)
end

rand(rng::AbstractRNG, b::LBPWNodeBelief) = rand(rng, b.dist)
state_mean(b::LBPWNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::LBPWNodeBelief) = b.o
POMDPs.history(b::LBPWNodeBelief) = tuple((a=b.a, o=b.o))


struct LBPWNodeFilter end

belief_type(::Type{LBPWNodeFilter}, ::Type{P}) where {P<:POMDP} = LBPWNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::LBPWNodeFilter, p::POMDP, s, a, sp, o, r) = LBPWNodeBelief(p, s, a, sp, o, r)

function push_weighted!(b::LBPWNodeBelief, ::LBPWNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
end

struct StateBelief{SRB<:LBPWNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::StateBelief) = first(rand(rng, b.sr_belief))
mean(b::StateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::StateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::StateBelief) = history(b.sr_belief)
