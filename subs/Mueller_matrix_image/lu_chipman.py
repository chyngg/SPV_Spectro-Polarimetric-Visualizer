import numpy as np

def calculate_params(mueller_matrix_image):
    eps = 1e-8

    m00 = mueller_matrix_image[:, :, 0, 0]
    m00_denom = np.maximum(m00, eps)[:, :, None, None]

    M = mueller_matrix_image / m00_denom
    m01 = M[:, :, 0, 1]
    m02 = M[:, :, 0, 2]
    m03 = M[:, :, 0, 3]
    diattenuation = np.sqrt(m01 ** 2 + m02 ** 2 + m03 ** 2)

    m10 = M[:, :, 1, 0]
    m20 = M[:, :, 2, 0]
    m30 = M[:, :, 3, 0]
    polarizance = np.sqrt(m10 ** 2 + m20 ** 2 + m30 ** 2)

    sum_sq = np.sum(M ** 2, axis=(2, 3))
    di_numerator = np.sqrt(np.maximum(sum_sq - 1, 0))
    depolarization_index = di_numerator / np.sqrt(3)

    m22 = M[:, :, 2, 2]
    m33 = M[:, :, 3, 3]
    m32 = M[:, :, 3, 2]
    m23 = M[:, :, 2, 3]

    val = np.sqrt((m22 + m33) ** 2 + (m32 - m23) ** 2) / 2.0
    val = np.clip(val, -1.0, 1.0)
    linear_retardance = np.arccos(val)

    return {
        "Depolarization Index": depolarization_index,
        "Diattenuation": diattenuation,
        "Polarizance": polarizance,
        "Linear Retardance": linear_retardance
    }


def get_decomposed_matrices(mueller_matrix_image):
    eps = 1e-8
    H, W, _, _ = mueller_matrix_image.shape

    m00 = mueller_matrix_image[:, :, 0, 0]
    m00_denom = np.maximum(m00, eps)[:, :, None, None]
    M = mueller_matrix_image / m00_denom

    D_vec = M[:, :, 0, 1:]  # (H, W, 3)
    D_sq = np.sum(D_vec ** 2, axis=2, keepdims=True)  # (H, W, 1)
    D_val = np.sqrt(D_sq)

    invalid_mask = D_sq > (1.0 - eps)
    scale = np.sqrt((1.0 - eps) / np.maximum(D_sq, eps))
    D_vec = np.where(invalid_mask, D_vec * scale, D_vec)
    D_sq = np.where(invalid_mask, 1.0 - eps, D_sq)

    a = np.sqrt(np.maximum(1 - D_sq, 0))

    M_D = np.zeros_like(M)
    M_D[:, :, 0, 0] = 1.0
    M_D[:, :, 0, 1:] = D_vec
    M_D[:, :, 1:, 0] = D_vec

    I = np.eye(3).reshape(1, 1, 3, 3)

    DD_T = D_vec[:, :, :, None] @ D_vec[:, :, None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        term2 = (1 - a[:, :, :, None]) * (DD_T / (D_sq[:, :, :, None] + eps))
        term2[D_sq[:, :, :, None].repeat(3, axis=2).repeat(3, axis=3) < eps] = 0

    m_D_sub = a[:, :, :, None] * I + term2
    M_D[:, :, 1:, 1:] = m_D_sub

    M_D_inv = M_D.copy()
    M_D_inv[:, :, 0, 1:] = -D_vec
    M_D_inv[:, :, 1:, 0] = -D_vec

    denom = np.maximum(1 - D_sq[:, :, :, None], eps)
    M_D_inv = M_D_inv / denom

    M_prime = M @ M_D_inv
    m_prime_3x3 = M_prime[:, :, 1:, 1:]  # (H, W, 3, 3)

    u, s, vt = np.linalg.svd(m_prime_3x3)

    # Rotation (m_R)
    m_R = u @ vt

    det = np.linalg.det(m_R)
    det = det[:, :, None, None]
    m_R = m_R * det

    # Depolarizer (m_Delta)
    m_Delta = u @ (np.expand_dims(s, axis=2) * np.eye(3).reshape(1, 1, 3, 3)) @ u.swapaxes(-1, -2)

    # Reconstruct 4x4 matrices
    M_R = np.zeros_like(M)
    M_R[:, :, 0, 0] = 1
    M_R[:, :, 1:, 1:] = m_R

    M_Delta = np.zeros_like(M)
    M_Delta[:, :, 0, 0] = 1
    M_Delta[:, :, 1:, 1:] = m_Delta

    return {
        "Matrix: Diattenuator": M_D,
        "Matrix: Retarder": M_R,
        "Matrix: Depolarizer": M_Delta,
        "Matrix: Corrected (M_Delta*M_R)": M_prime
    }