import numpy as np


def estimate_normalizing_transform(poses: np.ndarray) -> np.ndarray:
    """Estimate transform to normalize camera poses.

    Moves the average camera position to the origin and aligns the average
    down direction with world Y-axis.

    Args:
        poses: Camera poses with shape (N, 4, 4)
    Returns:
        4x4 transformation matrix
    """
    if len(poses) == 0:
        return np.eye(4)

    # Extract camera positions (translation vectors)
    positions = poses[:, :3, 3]  # Shape: (N, 3)
    avg_position = np.mean(positions, axis=0)

    # Extract down vectors (Y-axis) directly from all camera poses
    down_vectors = poses[:, :3, 1]  # Shape: (N, 3)

    # Compute average down direction
    avg_down = np.mean(down_vectors, axis=0)
    avg_down = avg_down / np.linalg.norm(avg_down)  # Normalize

    # Target down direction (world Y-axis)
    target_down = np.array([0, 1, 0])

    # Compute rotation to align avg_down with target_down
    # Using cross product and Rodrigues' rotation formula
    v = np.cross(avg_down, target_down)
    s = np.linalg.norm(v)
    c = np.dot(avg_down, target_down)

    if s < 1e-6:  # Vectors are already aligned
        rotation_matrix = np.eye(3)
    else:
        # Skew-symmetric matrix
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        # Rodrigues' rotation formula
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    # Apply rotation then translation
    transform[:3, 3] = -rotation_matrix @ avg_position

    return transform
