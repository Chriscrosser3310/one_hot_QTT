import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _mps_amplitude_on_bitstring(mps, bitstring):
    """
    Compute amplitude <bitstring|mps> by slicing each site tensor on its physical index
    and contracting along the chain (no dense conversion).
    Assumes computational basis bits '0'/'1'.
    """
    bits = [0 if c == "0" else 1 for c in bitstring]
    if hasattr(mps, "nsites"):
        L = mps.nsites
    else:
        L = len(mps.tensors)

    if len(bits) != L:
        raise ValueError(f"bitstring length {len(bits)} != mps length {L}")

    # Get per-site arrays in left-to-right order
    Ts = [np.asarray(T.data) for T in mps.tensors]

    # Each site tensor is typically shape (Dl, Dr, d) but boundaries may be rank-2.
    # We slice physical index, then multiply matrices.
    def slice_phys(A, b):
        if A.ndim == 3:      # (Dl, Dr, d)
            return A[:, :, b]
        elif A.ndim == 2:    # boundary maybe (Dr, d) or (Dl, d)
            # We interpret as (bond, d) -> slice gives (bond,)
            return A[:, b]
        else:
            raise ValueError(f"Unexpected tensor ndim={A.ndim}")

    # Initialize with first site
    M = slice_phys(Ts[0], bits[0])
    # M is either (Dr,) or (1,Dr) depending on boundary form; make it (1,Dr)
    if M.ndim == 1:
        M = M.reshape(1, -1)

    # Multiply through bulk
    for s in range(1, L - 1):
        A = slice_phys(Ts[s], bits[s])   # (Dl, Dr)
        M = M @ A

    if L == 1:
        # single-site state: amplitude is just scalar entry
        return M.reshape(-1)[0]

    # Last site
    last = slice_phys(Ts[-1], bits[-1])
    # last can be (Dl,) or (Dl,1); make it (Dl,1)
    if last.ndim == 1:
        last = last.reshape(-1, 1)

    amp = (M @ last)[0, 0]
    return amp


def mps_overlap_profile(mps, bs, D, value_mode="prob"):
    """
    Iterate over all tensor products of bitstrings in `bs` across D registers.

    Parameters
    ----------
    mps : quimb MPS
        State on L = D * q qubits (q inferred from len(bs[0])).
    bs : list[str]
        List of q-bit strings (each composed of '0'/'1'), e.g. one-hot basis strings.
    D : int
        Number of registers (tensor-product factors).

    Returns
    -------
    out : np.ndarray (complex)
        D-dimensional array of shape (B,)*D where B=len(bs).
        out[i0,i1,...,i_{D-1}] = amplitude of mps on bs[i0]||bs[i1]||...||bs[i_{D-1}]
        (concatenation order matches register order 0..D-1).
    """
    if D <= 0:
        raise ValueError("D must be positive.")
    if len(bs) == 0:
        raise ValueError("bs must be non-empty.")

    q = len(bs[0])
    if any(len(s) != q for s in bs):
        raise ValueError("All strings in bs must have the same length.")
    if any(set(s) - {"0", "1"} for s in bs):
        raise ValueError("All strings in bs must be binary strings of '0'/'1'.")

    # Check MPS length
    L_expected = D * q
    L_mps = mps.nsites if hasattr(mps, "nsites") else len(mps.tensors)
    if L_mps != L_expected:
        raise ValueError(f"MPS has {L_mps} sites, but D*q = {L_expected}.")

    B = len(bs)
    if value_mode == "prob" or value_mode == "abs":
        out = np.empty((B,) * D, dtype=float)
    else:
        out = np.empty((B,) * D, dtype=complex)

    for idx in np.ndindex(*((B,) * D)):
        full_bs = "".join(bs[i] for i in idx)

        #mps_bs = qtn.MPS_computational_state(full_bs)
        #out[idx] = (mps_bs.H @ mps) 
        if value_mode == "prob":
            out[idx] = np.abs(_mps_amplitude_on_bitstring(mps, full_bs))**2
        elif value_mode == "abs":
            out[idx] = np.abs(_mps_amplitude_on_bitstring(mps, full_bs))
        else:
            out[idx] = _mps_amplitude_on_bitstring(mps, full_bs)

    return out

def make_bond_dim_profile_and_gif_from_mps(
    mps_iter,
    mps_fn,
    fps=10,
    vmin=None,
    vmax=None,
    cmap="viridis",
    threshold=0.0,
    save_to_disk=True,
    folder="",
    fname="animation.gif"
):
    frames = []
    bond_dim_profile = []

    fig = plt.figure(figsize=(12, 5))

    for t, mps in enumerate(mps_iter):
        bond_sizes = mps.bond_sizes()
        bond_dim_profile.append(bond_sizes)

        data = np.asarray(mps_fn(mps))
        nd = data.ndim

        fig.clf()

        if nd == 1:
            ax = fig.add_subplot(111)
            y = np.asarray(data, dtype=float)
            x = np.arange(len(y))

            ax.plot(x, y)
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            ax.set_xlim(0, max(len(y) - 1, 1))
            ax.set_ylim((vmin, vmax))

        elif nd == 2:
            ax = fig.add_subplot(111)
            A = np.asarray(data, dtype=float)

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

        elif nd == 3:
            ax = fig.add_subplot(111, projection="3d")

            V = np.asarray(data)
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
                mask = mags >= threshold
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

        else:
            raise ValueError(f"mps_fn must return 1D/2D/3D array, got ndim={nd}.")

        plt.suptitle(f"t = {t}, bond dims = {bond_sizes}")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame.copy())

    plt.close(fig)

    import io
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", fps=fps)
    buf.seek(0)

    img = Image(data=buf.read(), format="gif")

    if save_to_disk:
        #with open(folder + "/animation.gif", "wb") as f:
        #    f.write(img.data) 

        imageio.mimsave(
            folder + fname,
            frames,        # list of numpy arrays or PIL Images
            fps=fps
        )

    return bond_dim_profile, img