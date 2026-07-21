import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Union




# TODO: give credit to the source
def golden_ratio():
    return (1 + np.sqrt(5)) / 2


def tetrahedron():
    return np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])


def octohedron():
    return np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ])


def cube():
    return np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])


def icosahedron():
    phi = golden_ratio()
    return np.array([
        [-1,  phi,  0],
        [-1, -phi,  0],
        [ 1,  phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]) / np.sqrt(1 + phi ** 2)


def dodecahedron():
    phi = golden_ratio()
    a, b = 1 / phi, 1 / (phi * phi)
    return np.array([
        [-a, -a,  b], [ a, -a,  b], [ a,  a,  b], [-a,  a,  b],
        [-a, -a, -b], [ a, -a, -b], [ a,  a, -b], [-a,  a, -b],
        [ b, -a, -a], [ b,  a, -a], [ b,  a,  a], [ b, -a,  a],
        [-b, -a, -a], [-b,  a, -a], [-b,  a,  a], [-b, -a,  a],
        [-a,  b, -a], [ a,  b, -a], [ a,  b,  a], [-a,  b,  a],
    ]) / np.sqrt(a ** 2 + b ** 2)


def standard(n=8, elevation=15):
    """
    """
    pphi =  elevation * np.pi / 180
    nphi = -elevation * np.pi / 180
    coords = []
    for phi in [pphi, nphi]:
        for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
            coords.append([
                np.cos(theta) * np.cos(phi),
                np.sin(phi),
                np.sin(theta) * np.cos(phi),
            ])
    coords.append([0,  0,  1])
    coords.append([0,  0, -1])
    return np.array(coords)


def swirl(n=120, cycles=1, elevation_range=(-45, 60)):
    """
    """
    pphi = elevation_range[0] * np.pi / 180
    nphi = elevation_range[1] * np.pi / 180
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = []
    for i, phi in enumerate(np.linspace(pphi, nphi, n)):
        coords.append([
            np.cos(cycles * thetas[i]) * np.cos(phi),
            np.sin(phi),
            np.sin(cycles * thetas[i]) * np.cos(phi),
        ])
    return np.array(coords)



HomogeneousTransform = Annotated[
    Union[np.ndarray, torch.Tensor],
    "Shape: (..., 4, 4) - batch dimensions followed by 4x4 homogeneous transformation matrix"
]


def matrix3x4_to_4x4(matrix3x4: HomogeneousTransform) -> HomogeneousTransform:
    """
    Convert a 3x4 transformation matrix to a 4x4 transformation matrix.
    """
    bottom = torch.zeros_like(matrix3x4[:, 0, :].unsqueeze(-2))
    bottom[..., -1] = 1
    return torch.cat([matrix3x4, bottom], dim=-2)


def view_matrix(
    camera_position: torch.Tensor,
    lookat_position: torch.Tensor = torch.tensor([0, 0, 0]),
    up             : torch.Tensor      = torch.tensor([0, 1, 0]),
) -> HomogeneousTransform:
    """
    Given lookat position, camera position, and up vector, compute cam2world poses.
    """
    if camera_position.ndim == 1:
        camera_position = camera_position.unsqueeze(0)
    if lookat_position.ndim == 1:
        lookat_position = lookat_position.unsqueeze(0)
    camera_position = camera_position.float()
    lookat_position = lookat_position.float()

    cam_u = up.unsqueeze(0).repeat(len(lookat_position), 1).float().to(camera_position.device)

    # handle degenerate cases
    crossp = torch.abs(torch.cross(lookat_position - camera_position, cam_u, dim=-1)).max(dim=-1).values
    camera_position[crossp < 1e-6] += 1e-6

    cam_z = F.normalize((lookat_position - camera_position), dim=-1)
    cam_x = F.normalize(torch.cross(cam_z, cam_u, dim=-1), dim=-1)
    cam_y = F.normalize(torch.cross(cam_x, cam_z, dim=-1), dim=-1)
    poses = torch.stack([cam_x, cam_y, -cam_z, camera_position], dim=-1) # same as nerfstudio convention [right, up, -lookat]
    poses = matrix3x4_to_4x4(poses)
    return poses


def sample_view_matrices(n: int, radius: float, lookat_position: torch.Tensor=torch.tensor([0, 0, 0])) -> HomogeneousTransform:
    """
    Sample n uniformly distributed view matrices spherically with given radius.
    """
    tht = torch.rand(n) * torch.pi * 2
    phi = torch.rand(n) * torch.pi
    world_x = radius * torch.sin(phi) * torch.cos(tht)
    world_y = radius * torch.sin(phi) * torch.sin(tht)
    world_z = radius * torch.cos(phi)
    camera_position = torch.stack([world_x, world_y, world_z], dim=-1)
    lookat_position = lookat_position.unsqueeze(0).repeat(n, 1)
    return view_matrix(
        camera_position.to(lookat_position.device),
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device)
    )


def sample_view_matrices_polyhedra(polygon: str, radius: float, lookat_position: torch.Tensor=torch.tensor([0, 0, 0]), **kwargs) -> HomogeneousTransform:
    """
    Sample view matrices according to a polygon with given radius.
    """
    camera_position = torch.from_numpy(eval(polygon)(**kwargs)) * radius
    return view_matrix(
        camera_position.to(lookat_position.device) + lookat_position,
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device)
    )