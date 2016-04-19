module MWM

# this is an implementation of the MWM building block used in the paper

using Base.Test

export Data, a, update!, sample, run!

type Data
    x::UnitRange{Int}
    w::Vector{Float64}
    ϵ::Vector{Float64}
end

import Base.deepcopy
Base.deepcopy(mwm::Data) = Data(deepcopy(mwm.x), deepcopy(mwm.w), deepcopy(mwm.ϵ))

Data(n::Int, ϵ::Float64) = Data(1:n, fill(1/n, n), fill(ϵ, n))
Data(n::Int, ϵ::Vector{Float64}) = Data(1:n, fill(1/n, n), ϵ)

function a(mwm::Data)
    mwm.w .* mwm.ϵ / dot(mwm.w, mwm.ϵ)
end

function update!(mwm::Data, loss::Vector{Float64})
    lossa = dot(a(mwm), loss)
    # the '+' in the paper should be a '-' like here
    mwm.w = mwm.w .* (1 - mwm.ϵ .* (loss - lossa))
end

function sample(mwm::Data)
    dist = mwm.w .* mwm.ϵ
    psum = cumsum(dist)
    tot = psum[end]
    r = rand()*tot
    mwm.x[searchsortedfirst(psum, r)]
end

# check whether the guarantee holds (don't know what constant is hidden by big O
# so try to estimate empirically using ratio of sum1 and sum2)
function run!(mwm::Data, losses::Vector{Vector{Float64}})
    w0 = copy(mwm.w)
    sum1 = zeros(length(mwm.x))
    sum2 = zeros(length(mwm.x))
    for l = losses
        la = dot(l, a(mwm))
        sum1 += l - la + mwm.ϵ .* (la - l).^2
        sum2 += -log(w0)./mwm.ϵ
        update!(mwm, l)
    end
    sum1 + sum2, sum1, sum2
end

function gen(n::Int, t::Int)
    losses = Vector{Float64}[]
    for i = 1:t
        push!(losses, 2*rand(n)-1)
    end

    w = rand(n)
    ϵ = 1/2*rand(n)

    Data(1:n, w, ϵ), losses
end

# scratch code for estimating the constant
best = (0.0, MWM.Data(1,0.0), Vector{Float64}[], 0.0, 0.0)
for i = 1:10000
    n = rand(1:100)
    t = rand(1:200)
    mwm, losses = gen(n, t)
    old = deepcopy(mwm)
    res, s1, s2 = run!(mwm, losses)
    ok = all(res .>= 0)
    m = maximum(-s1./s2)
    if m > best[1]
        best = (m, old, losses, s1, s2)
    end
    #if !ok
    #    println(n, " ", t)
    #    println(w0, " ", ϵ0)
    #    println(res)
    #end
    #@test ok
end
println(best)

end
