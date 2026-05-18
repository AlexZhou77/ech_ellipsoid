import numpy as np
import math

class Ellipsoid:
    def __init__(self, a=1.0, b=np.sqrt(2.0)):
        self.a = float(a)
        self.b = float(b)
        # The default pole used for stereographic projection. 
        self.pole_eta = math.pi / 4
        self.pole_theta1 = 0.0
        self.pole_theta2 = 0.0

    def ellipsoid_point(self, eta, theta1, theta2):
        eta, theta1, theta2 = np.broadcast_arrays(eta, theta1, theta2)

        r1 = np.sqrt(self.a / np.pi) * np.cos(eta)
        r2 = np.sqrt(self.b / np.pi) * np.sin(eta)

        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)

        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)

        return np.stack([x1, y1, x2, y2], axis=-1)
    
    # Normalize a vector
    def _normalize(self, v):
        v = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("Cannot normalize the zero vector.")
        return v / norm

    # Given p in S^3, find an orthonormal basis of the tangent space at p
    def _orthonormal_basis_perp_to(self, p):
        p = self._normalize(p)
        basis = []

        for e in np.eye(4):
            w = e - np.dot(e, p) * p

            for u in basis:
                w -= np.dot(w, u) * u

            norm = np.linalg.norm(w)
            if norm > 1e-10:
                basis.append(w / norm)

            if len(basis) == 3:
                return np.array(basis)

        raise ValueError("Could not construct a 3D orthonormal basis orthogonal to p.")
     

    def _ellipsoid_to_s3(self, x):
        x = np.asarray(x, dtype=float)
        scale = np.array([
            np.sqrt(np.pi / self.a),
            np.sqrt(np.pi / self.a),
            np.sqrt(np.pi / self.b),
            np.sqrt(np.pi / self.b),
        ])
        return x * scale

    # Stereographic projection from p in S^3 to R^3
    def _stereographic_projection_s3(self, z, q):
        z = np.asarray(z, dtype=float)
        q = self._normalize(q)

        dot = np.sum(z * q, axis=-1, keepdims=True)
        denom = 1.0 - dot

        valid = np.abs(denom) >= 1e-10
        projected = np.full_like(z, np.nan, dtype=float)
        projected[valid[..., 0]] = (
            (z - dot * q)[valid[..., 0]] / denom[valid[..., 0]]
        )

        basis = self._orthonormal_basis_perp_to(q)
        coords_3d = projected @ basis.T

        return coords_3d
    
    def stereographic_projection_ellipsoid(self, x):
        p = self.ellipsoid_point(self.pole_eta, self.pole_theta1, self.pole_theta2)
        p = self._ellipsoid_to_s3(p)
        z = self._ellipsoid_to_s3(x)
        return self._stereographic_projection_s3(z, p)
    
    def torus_surface(self, eta, n1=800, n2=800):
        theta1 = np.linspace(0, 2 * np.pi, n1)
        theta2 = np.linspace(0, 2 * np.pi, n2)
        TH1, TH2 = np.meshgrid(theta1, theta2, indexing="ij")
        x = self.ellipsoid_point(eta, TH1, TH2)
        w = self.stereographic_projection_ellipsoid(x)
        return w[..., 0], w[..., 1], w[..., 2]

    def theta1_circle(self, eta, theta1, n=800):
        theta2 = np.linspace(0, 2 * np.pi, n)
        x = self.ellipsoid_point(eta, theta1, theta2)
        w = self.stereographic_projection_ellipsoid(x)
        return w[:, 0], w[:, 1], w[:, 2]

    def theta2_circle(self, eta, theta2, n=800):
        theta1 = np.linspace(0, 2 * np.pi, n)
        x = self.ellipsoid_point(eta, theta1, theta2)
        w = self.stereographic_projection_ellipsoid(x)
        return w[:, 0], w[:, 1], w[:, 2]

    def braid_critical_eta(self, m1, m2):
        m1 = int(m1)
        m2 = int(m2)
        if m1 <= 0 or m2 <= 0:
            raise ValueError("m1 and m2 must be positive integers.")
        return math.atan(math.sqrt(m1 / m2))

    def braid_s_from_eta(self, eta, m1, m2):
        eta = np.asarray(eta, dtype=float)
        eta_star = self.braid_critical_eta(m1, m2)
        cos_eta = np.clip(np.cos(eta), 1e-15, None)
        sin_eta = np.clip(np.sin(eta), 1e-15, None)
        cos_star = math.cos(eta_star)
        sin_star = math.sin(eta_star)
        return 2.0 * (m2 * np.log(cos_star / cos_eta) + m1 * np.log(sin_star / sin_eta)) / (m1 + m2)

    def braid_etas(self, s, m1, m2, tol=1e-9, max_iter=80):
        s = float(s)
        if s < -tol:
            raise ValueError("s must be nonnegative.")

        eta_star = self.braid_critical_eta(m1, m2)
        if abs(s) <= tol:
            return (eta_star,)

        eps = 1e-8

        def branch_value(eta):
            return float(self.braid_s_from_eta(eta, m1, m2))

        left_low, right_low = eps, eta_star - eps
        for _ in range(max_iter):
            mid = 0.5 * (left_low + right_low)
            if branch_value(mid) > s:
                left_low = mid
            else:
                right_low = mid
        eta_low = 0.5 * (left_low + right_low)

        left_high, right_high = eta_star + eps, 0.5 * np.pi - eps
        for _ in range(max_iter):
            mid = 0.5 * (left_high + right_high)
            if branch_value(mid) < s:
                left_high = mid
            else:
                right_high = mid
        eta_high = 0.5 * (left_high + right_high)

        return eta_low, eta_high

    def braid_slice_branches(self, s, m1, m2, phase=0.0, n=500):
        etas = self.braid_etas(s, m1, m2)
        gcd = math.gcd(int(m1), int(m2))
        p = int(m1) // gcd
        q = int(m2) // gcd
        t = np.linspace(0, 2 * np.pi, n)
        branches = []

        for eta in etas:
            branch_x = []
            branch_y = []
            branch_z = []
            for k in range(gcd):
                theta1 = p * t
                theta2 = -q * t + (phase + 2 * np.pi * k) / int(m1)
                x = self.ellipsoid_point(eta, theta1, theta2)
                w = self.stereographic_projection_ellipsoid(x)
                branch_x.extend(w[:, 0].tolist())
                branch_y.extend(w[:, 1].tolist())
                branch_z.extend(w[:, 2].tolist())
                branch_x.append(np.nan)
                branch_y.append(np.nan)
                branch_z.append(np.nan)
            branches.append((np.array(branch_x), np.array(branch_y), np.array(branch_z)))

        return branches
    
    def gamma1(self, n=800):
        theta1 = np.linspace(0, 2 * np.pi, n)
        x = self.ellipsoid_point(0.0, theta1, 0.0)
        w = self.stereographic_projection_ellipsoid(x)
        return w[:, 0], w[:, 1], w[:, 2]
    
    def gamma2(self, n=800):
        theta2 = np.linspace(0, 2 * np.pi, n)
        x = self.ellipsoid_point(np.pi / 2, 0.0, theta2)
        w = self.stereographic_projection_ellipsoid(x)
        return w[:, 0], w[:, 1], w[:, 2]

    def gamma1_contact_plane_patches(self, n=14, eta_offset=0.09, scale=0.11):
        theta1 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

        base = self.ellipsoid_point(0.0, theta1, 0.0)
        basis1 = self.ellipsoid_point(eta_offset, theta1, 0.0)
        basis2 = self.ellipsoid_point(eta_offset, theta1, 0.5 * np.pi)

        base_proj = self.stereographic_projection_ellipsoid(base)
        basis1_proj = self.stereographic_projection_ellipsoid(basis1)
        basis2_proj = self.stereographic_projection_ellipsoid(basis2)

        vec1 = basis1_proj - base_proj
        vec2 = basis2_proj - base_proj

        vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
        vec1_norm = np.clip(vec1_norm, 1e-12, None)
        unit_vec1 = vec1 / vec1_norm

        vec2 = vec2 - np.sum(vec2 * unit_vec1, axis=1, keepdims=True) * unit_vec1
        vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
        vec2_norm = np.clip(vec2_norm, 1e-12, None)
        unit_vec2 = vec2 / vec2_norm

        patch_vertices = []
        outline_x = []
        outline_y = []
        outline_z = []
        tri_i = []
        tri_j = []
        tri_k = []

        for index, (center, dir1, dir2) in enumerate(zip(base_proj, unit_vec1, unit_vec2)):
            corner0 = center + scale * (dir1 + dir2)
            corner1 = center + scale * (dir1 - dir2)
            corner2 = center + scale * (-dir1 - dir2)
            corner3 = center + scale * (-dir1 + dir2)
            corners = np.array([corner0, corner1, corner2, corner3])

            start = 4 * index
            patch_vertices.extend(corners.tolist())
            tri_i.extend([start, start])
            tri_j.extend([start + 1, start + 2])
            tri_k.extend([start + 2, start + 3])

            loop = np.vstack([corners, corners[0]])
            outline_x.extend(loop[:, 0].tolist() + [np.nan])
            outline_y.extend(loop[:, 1].tolist() + [np.nan])
            outline_z.extend(loop[:, 2].tolist() + [np.nan])

        patch_vertices = np.array(patch_vertices)
        return (
            patch_vertices[:, 0],
            patch_vertices[:, 1],
            patch_vertices[:, 2],
            np.array(tri_i),
            np.array(tri_j),
            np.array(tri_k),
            np.array(outline_x),
            np.array(outline_y),
            np.array(outline_z),
        )

    def gamma2_contact_plane_patches(self, n=14, eta_offset=0.09, scale=0.11):
        theta2 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

        base = self.ellipsoid_point(0.5 * np.pi, 0.0, theta2)
        basis1 = self.ellipsoid_point(0.5 * np.pi - eta_offset, 0.0, theta2)
        basis2 = self.ellipsoid_point(0.5 * np.pi - eta_offset, 0.5 * np.pi, theta2)

        base_proj = self.stereographic_projection_ellipsoid(base)
        basis1_proj = self.stereographic_projection_ellipsoid(basis1)
        basis2_proj = self.stereographic_projection_ellipsoid(basis2)

        vec1 = basis1_proj - base_proj
        vec2 = basis2_proj - base_proj

        vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
        vec1_norm = np.clip(vec1_norm, 1e-12, None)
        unit_vec1 = vec1 / vec1_norm

        vec2 = vec2 - np.sum(vec2 * unit_vec1, axis=1, keepdims=True) * unit_vec1
        vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
        vec2_norm = np.clip(vec2_norm, 1e-12, None)
        unit_vec2 = vec2 / vec2_norm

        patch_vertices = []
        outline_x = []
        outline_y = []
        outline_z = []
        tri_i = []
        tri_j = []
        tri_k = []

        for index, (center, dir1, dir2) in enumerate(zip(base_proj, unit_vec1, unit_vec2)):
            corner0 = center + scale * (dir1 + dir2)
            corner1 = center + scale * (dir1 - dir2)
            corner2 = center + scale * (-dir1 - dir2)
            corner3 = center + scale * (-dir1 + dir2)
            corners = np.array([corner0, corner1, corner2, corner3])

            start = 4 * index
            patch_vertices.extend(corners.tolist())
            tri_i.extend([start, start])
            tri_j.extend([start + 1, start + 2])
            tri_k.extend([start + 2, start + 3])

            loop = np.vstack([corners, corners[0]])
            outline_x.extend(loop[:, 0].tolist() + [np.nan])
            outline_y.extend(loop[:, 1].tolist() + [np.nan])
            outline_z.extend(loop[:, 2].tolist() + [np.nan])

        patch_vertices = np.array(patch_vertices)
        return (
            patch_vertices[:, 0],
            patch_vertices[:, 1],
            patch_vertices[:, 2],
            np.array(tri_i),
            np.array(tri_j),
            np.array(tri_k),
            np.array(outline_x),
            np.array(outline_y),
            np.array(outline_z),
        )

    def reeb_trajectory(self, eta, theta1_0=0.0, theta2_0=0.0, T=12.0, N=4000):
        """
        The Reeb flow lines are given by the equations:
            theta1(t) = theta1_0 + 2*pi*t/a,
            theta2(t) = theta2_0 + 2*pi*t/b.
        """
        t = np.linspace(0, T, N)

        theta1 = theta1_0 + 2 * np.pi * t / self.a
        theta2 = theta2_0 + 2 * np.pi * t / self.b

        x = self.ellipsoid_point(eta, theta1,theta2)

        w = self.stereographic_projection_ellipsoid(x)

        return w[:, 0], w[:, 1], w[:, 2]