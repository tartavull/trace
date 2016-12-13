segnet_params = {
    # downsampling sizes
    'd1': [
        [4, 4, 1, 4],
        [4, 4, 4, 4]
    ],
    'd2': [
        [4, 4, 4, 8],
        [4, 4, 8, 8]
    ],
    'd3': [
        [4, 4, 8, 16],
        [4, 4, 16, 16],
        [4, 4, 16, 16]
    ],
    'd4': [
        [4, 4, 16, 32],
        [4, 4, 32, 32],
        [4, 4, 32, 32]
    ],
    'd5': [
        [4, 4, 32, 64],
        [4, 4, 64, 64],
        [4, 4, 64, 64]
    ],

    # upsampling sizes
    'u1': [
        [4, 4, 64, 64],
        [4, 8, 32, 32],
        [4, 8, 32, 32],
        [4, 8, 32, 32],
    ],
    'u2': [
        [4, 4, 32, 32],
        [4, 8, 16, 16],
        [4, 8, 16, 16],
        [4, 8, 16, 16]
    ],
    'u3': [
        [4, 8, 16, 16],
        [4, 16, 8, 8],
        [4, 16, 8, 8],
        [4, 16, 8, 8]
    ],
    'u4': [
        [4, 16, 8, 8],
        [4, 32, 4, 4],
        [4, 32, 4, 4],
        [4, 32, 4, 4]
    ],
    'u5': [
        [4, 32, 4, 4],
        [4, 64, 2, 2],
        [4, 64, 2, 2],
        [4, 64, 2, 2]
    ],

    'fc': 200,
    'lr': 0.001,
    'out': 101
}