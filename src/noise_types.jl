abstract type NoiseChannel end

struct NoNoise <: NoiseChannel
    p::Real
end
struct AmplitudeDamping <: NoiseChannel
    p::Real
end
struct PhaseDamping <: NoiseChannel
    p::Real
end
struct PhaseFlip <: NoiseChannel
    p::Real
end
struct BitFlip <: NoiseChannel
    p::Real
end
struct BitPhaseFlip <: NoiseChannel
    p::Real
end
struct DepolarizingChannel <: NoiseChannel
    p::Real
end

NoNoise() = NoNoise(0.0)
