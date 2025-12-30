import numpy as np
import matplotlib.pyplot as plt
from one_hot_basis import one_hot_bounce_bitstrings

def disorder_landscape(D, n, q, W, disorder_type,
                seed=None,
                vmin=None,
                vmax=None,
                cmap="viridis",
                threshold=0):
    
    rng = np.random.default_rng(seed=seed)

    def Z_potential(bitstring):
        total = 0
        for i in range(len(bitstring)):
            Si = 1 - 2*int(bitstring[i])
            coeff = W*np.pi*(2*rng.random()-1.)
            total += coeff * Si
        return total

    def ZZ_potential(bitstring):
        total = 0
        for i in range(len(bitstring) - 1):
            Si = 1 - 2*int(bitstring[i])
            Sip1 = 1 - 2*int(bitstring[i+1])
            coeff = W*np.pi*(2*rng.random()-1.)
            total += coeff * Si * Sip1
        return total
    
    if disorder_type == "ZZ":
        disorder_potential = ZZ_potential
    elif disorder_type == "Z":
        disorder_potential = Z_potential

    basis_list = one_hot_bounce_bitstrings(n, q)
    if D == 2:
        A = np.zeros((len(basis_list), len(basis_list)))
        for i, x in enumerate(basis_list):
            for j, y in enumerate(basis_list):
                bs = x + y
                A[i, j] = disorder_potential(bs)

        fig = plt.figure(figsize=(6, 5))

        ax = fig.add_subplot(111)

        im = ax.imshow(
            A,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_xlabel("x") 
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    elif D == 3:
        V = np.zeros((len(basis_list), len(basis_list), len(basis_list)))
        for i, x in enumerate(basis_list):
            for j, y in enumerate(basis_list):
                for k, z in enumerate(basis_list):
                    bs = x + y + z
                    V[i, j, k] = disorder_potential(bs)
        
        fig = plt.figure(figsize=(6, 5))
        
        ax = fig.add_subplot(111, projection="3d")

        nx, ny, nz = V.shape

        xs, ys, zs = np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            np.arange(nz),
            indexing="ij",
        )

        vals = V.ravel()
        xs = xs.ravel()
        ys = ys.ravel()
        zs = zs.ravel()

        mags = np.abs(vals)

        # optional hard threshold (still useful for performance)
        if threshold is not None and threshold > 0:
            mask = mags >= 0
            xs, ys, zs, vals, mags = (
                xs[mask], ys[mask], zs[mask], vals[mask], mags[mask]
            )

        if mags.size == 0:
            # nothing to plot
            pass
        else:
            # ---- normalize magnitudes ----
            mags_norm = mags / mags.max()

            # ---- size mapping ----
            size_min = 0.0
            size_max = 200.0
            sizes = size_min + (size_max - size_min) * mags_norm

            # ---- alpha mapping (fade near zero) ----
            alpha_min = 0.05     # faint but visible
            alpha_max = 0.9
            alphas = alpha_min + (alpha_max - alpha_min) * mags_norm

            # Matplotlib expects a single alpha OR per-point RGBA,
            # so we encode alpha into RGBA colors explicitly.
            cmap_obj = plt.get_cmap(cmap)
            colors = cmap_obj(
                (vals - (vmin if vmin is not None else vals.min())) /
                ((vmax if vmax is not None else vals.max()) -
                (vmin if vmin is not None else vals.min()) + 1e-12)
            )
            colors[:, 3] = alphas  # overwrite alpha channel

            sc = ax.scatter(
                xs,
                ys,
                zs,
                c=colors,
                s=sizes,
                depthshade=True,
            )

        # ---- fixed axes ----
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_zlim(-0.5, nz - 0.5)

        ax.set_xticks(np.arange(nx))
        ax.set_yticks(np.arange(ny))
        ax.set_zticks(np.arange(nz))

        ax.view_init(elev=20, azim=-60)

        try:
            ax.set_box_aspect((nx, ny, nz))
        except Exception:
            pass

        # colorbar still reflects value (ignores alpha)
        if mags.size > 0:
            mappable = plt.cm.ScalarMappable(cmap=cmap)
            mappable.set_array(vals)
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

#disorder_landscape(3, 2, 8, 1, "ZZ", seed=None)