# Here define the noise models and their Kraus operators
"""
    NoiseChannel

Abstract supertype for different noise models.
Each model is described by its Kraus operators,
defined as in Ch. 8 of Nielsen&Chuang textbook.
"""

include("noise_types.jl")
include("unitary_parametrizations.jl")

KrausOps = Vector{Matrix{T}} where {T<:Number}

kraus_ops(::NoNoise) = Matrix{Int32}[]

kraus_ops(noise::AmplitudeDamping) = [
    [
        1 0
        0 sqrt(1 - noise.p)
    ],
    [
        0 sqrt(noise.p)
        0 0
    ],
]

kraus_ops(noise::PhaseDamping) = [
    [
        1 0
        0 sqrt(1 - noise.p)
    ],
    [
        0 0
        0 sqrt(noise.p)
    ],
]

kraus_ops(noise::PhaseFlip) = [sqrt(noise.p) * [
        1 0
        0 -1
    ], sqrt(1 - noise.p) * [
        1 0
        0 1
    ]]

kraus_ops(noise::BitFlip) = [sqrt(noise.p) * [
        0 1
        1 0
    ], sqrt(1 - noise.p) * [
        1 0
        0 1
    ]]

kraus_ops(noise::BitPhaseFlip) = [sqrt(noise.p) * [
        0 -im
        im 0
    ], sqrt(1 - noise.p) * [
        1 0
        0 1
    ]]

function kraus_ops(noise::DepolarizingChannel)
    return [
        sqrt(1 - 3 * noise.p / 4) * [
            1 0
            0 1
        ],
        sqrt(noise.p / 4) * [
            0 1
            1 0
        ],
        sqrt(noise.p / 4) * [
            0 -im
            im 0
        ],
        sqrt(noise.p / 4) * [
            1 0
            0 -1
        ],
    ]
end
