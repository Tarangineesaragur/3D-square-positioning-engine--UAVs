# 3D-square-positioning-engine--UAVs
import math
import random

ANCHORS = [
    (0.0, 0.0, 0.0),
    (100.0, 0.0, 0.0),
    (0.0, 100.0, 0.0),
    (0.0, 0.0, 100.0),
    (50.0, 50.0, 50.0)
]

TRUE_POSITION = (25.0, 35.0, 40.0)

NOISE_STD_DEV = 2.0 


def get_matrix_transpose(A):
    rows = len(A)
    cols = len(A[0])
    A_T = [[A[r][c] for r in range(rows)] for c in range(cols)]
    return A_T

def get_matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    return C

def get_matrix_inverse_3x3(M):
    if len(M) != 3 or len(M[0]) != 3:
        raise ValueError("Inverse function only handles 3x3 matrices.")

    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    inv_det = 1.0 / det
    
    inv = [
        [
            (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * inv_det,
            (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * inv_det,
            (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * inv_det
        ],
        [
            (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * inv_det,
            (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * inv_det,
            (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * inv_det
        ],
        [
            (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * inv_det,
            (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * inv_det,
            (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * inv_det
        ]
    ]

    return inv


def euclidean_distance_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def generate_noisy_measurements(true_position, anchors, noise_std_dev):
    distances = []
    for anchor in anchors:
        true_dist = euclidean_distance_3d(true_position, anchor)
        noisy_dist = true_dist + random.gauss(0, noise_std_dev)
        distances.append(noisy_dist)
    return distances

def solve_3d_least_squares(anchors, distances):
    N = len(anchors)
    if N < 4:
        raise ValueError("Least Squares requires at least 4 anchors for 3D.")

    x1, y1, z1 = anchors[0]
    r1 = distances[0]
    P1 = x1**2 + y1**2 + z1**2

    A = []
    b = []

    for i in range(1, N):
        xi, yi, zi = anchors[i]
        ri = distances[i]
        Pi = xi**2 + yi**2 + zi**2

        A_row = [
            2.0 * (x1 - xi),
            2.0 * (y1 - yi),
            2.0 * (z1 - zi)
        ]
        A.append(A_row)

        b_val = [ri**2 - r1**2 - Pi + P1]
        b.append(b_val)

    A_T = get_matrix_transpose(A)

    ATA = get_matrix_multiply(A_T, A)

    try:
        ATA_inv = get_matrix_inverse_3x3(ATA)
    except ValueError as e:
        print(f"Error in matrix inversion: {e}")
        return None

    ATb = get_matrix_multiply(A_T, b)

    solution_matrix = get_matrix_multiply(ATA_inv, ATb)

    x_est = solution_matrix[0][0]
    y_est = solution_matrix[1][0]
    z_est = solution_matrix[2][0]

    return x_est, y_est, z_est


if __name__ == "__main__":
    print("--- 3D Least Squares Positioning Engine ---")
    print(f"Anchors Used (N={len(ANCHORS)}): {ANCHORS}")
    print(f"True Position: {TRUE_POSITION}")
    print(f"Noise Std Dev: {NOISE_STD_DEV} m")

    distances = generate_noisy_measurements(
        TRUE_POSITION, ANCHORS, NOISE_STD_DEV
    )
    
    print("\n--- SIMULATED INPUT ---")
    true_distances = [
        euclidean_distance_3d(TRUE_POSITION, a)
        for a in ANCHORS
    ]
    print(f"True Distances (r_t): {[f'{d:.2f}m' for d in true_distances]}")
    print(f"Noisy Measurements (r_n): {[f'{d:.2f}m' for d in distances]}")


    estimated_position = solve_3d_least_squares(ANCHORS, distances)

    print("\n--- LEAST SQUARES ESTIMATION RESULT ---")
    if estimated_position:
        est_x, est_y, est_z = estimated_position
        print(f"Estimated Position (x, y, z): ({est_x:.2f} m, {est_y:.2f} m, {est_z:.2f} m)")

        error = euclidean_distance_3d(estimated_position, TRUE_POSITION)

        print(f"\nTotal Estimation Error: {error:.2f} m")
    else:
        print("Estimation failed due to singular matrix or calculation error.")
