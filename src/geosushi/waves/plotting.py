
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


def create_wave_energy_colormap():
    """
    Create a custom colormap for wave spectral energy density E(f,θ) [m²·s·rad⁻¹].
    Colors scale proportionally from vmin (ice-white) to vmax (violet).
    
    Color scheme (low → high energy density):
    - Branco gelo → Cinza → Azul → Verde → Amarelo → Laranja → Vermelho → Violeta
    """
    colors = [
        '#F8F8FF',  # Branco gelo (alice blue)
        '#E0E0E0',  # Cinza claro
        '#4A90E2',  # Azul
        '#50C878',  # Verde esmeralda
        '#FFD700',  # Amarelo ouro
        '#FF8C00',  # Laranja escuro
        '#FF0000',  # Vermelho
        '#8B008B',  # Violeta escuro
    ]
    
    # Create smooth transitions between colors
    n_bins = 256
    cmap = mpl.colors.LinearSegmentedColormap.from_list('wave_energy', colors, N=n_bins)
    return cmap


def plot_directional_spectrum(E2d, freq, dirs, selected_time=None, hs=None, tp=None, dp=None,
                              vmin=None, vmax=None, n_levels=100, partitions=None):
    """
    Plot 2D directional spectrum in polar coordinates.

    Parameters:
    -----------
    E2d : ndarray
        Spectrum directional 2D [m²·s·rad⁻¹]
    freq : ndarray
        Array of frequencies [Hz]
    dirs : ndarray
        Array of directions [degrees]
    selected_time : datetime, optional
        Timestamp of the data
    hs : float, optional
        Significant wave height [m] (used only when partitions is None)
    tp : float, optional
        Peak period [s] (used only when partitions is None)
    dp : float, optional
        Peak direction [degrees] (used only when partitions is None)
    partitions : list of dict, optional
        List of wave systems, each with keys 'Hs', 'Tp', 'Dp'.
        When provided, individual systems are shown instead of total.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    Eplot = np.nan_to_num(E2d, nan=0.0, neginf=0.0, posinf=0.0)

    # Garantir arrays 1D
    freq_plot = np.asarray(freq).flatten()
    dirs_plot = np.asarray(dirs).flatten()

    # Convert directions for radians e ordenar
    dirs_rad_plot = np.radians(dirs_plot)
    sort_idx = np.argsort(dirs_rad_plot)
    dirs_sorted = dirs_rad_plot[sort_idx]
    Eplot_sorted = Eplot[:, sort_idx]

    # Ensure periodic continuity (0 to 2π)
    if not np.isclose(dirs_sorted[0], 0.0):
        dirs_sorted = np.insert(dirs_sorted, 0, 0.0)
        Eplot_sorted = np.insert(Eplot_sorted, 0, Eplot_sorted[:, 0], axis=1)
    if not np.isclose(dirs_sorted[-1], 2*np.pi):
        dirs_sorted = np.append(dirs_sorted, 2*np.pi)
        Eplot_sorted = np.concatenate([Eplot_sorted, Eplot_sorted[:, 0:1]], axis=1)

    # Eixo radial = period (s)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.where(freq_plot > 0, 1.0 / freq_plot, 0)

    theta, r = np.meshgrid(dirs_sorted, period)

    # Use fixed vmin/vmax if provided, otherwise span the full data range.
    # vmin starts at 0 so ice-white correctly represents near-zero energy regions.
    data_max = np.nanmax(Eplot_sorted)
    if data_max <= 0:
        data_max = 1.0
    _vmin = vmin if vmin is not None else 0.0
    _vmax = vmax if vmax is not None else data_max
    levels = np.linspace(_vmin, _vmax, n_levels)

    # Interpolate to finer grid using bicubic spline on the regular grid
    # Remove duplicate angular endpoint (0° == 360°) to avoid redundancy in spline
    dirs_nodupe = dirs_sorted[:-1]
    Eplot_nodupe = Eplot_sorted[:, :-1]

    # Use only valid (non-zero) periods
    valid_freq = period > 0
    period_valid = period[valid_freq]
    Eplot_valid = Eplot_nodupe[valid_freq, :]

    # Sort by period ascending (required by RectBivariateSpline)
    sort_r = np.argsort(period_valid)
    period_asc = period_valid[sort_r]
    E_asc = Eplot_valid[sort_r, :]

    # Bicubic spline interpolation on regular grid
    # Pad angular direction with wrap-around data to avoid extrapolation artifacts near 0/2π
    n_pad = max(4, len(dirs_nodupe) // 6)
    dirs_padded = np.concatenate([
        dirs_nodupe[-n_pad:] - 2*np.pi,
        dirs_nodupe,
        dirs_nodupe[:n_pad] + 2*np.pi,
    ])
    E_padded = np.hstack([E_asc[:, -n_pad:], E_asc, E_asc[:, :n_pad]])
    spl = RectBivariateSpline(period_asc, dirs_padded, E_padded, kx=3, ky=3)

    # Fine grid - endpoint=False keeps angular grid truly periodic
    n_theta_fine = len(dirs_nodupe) * 4
    n_r_fine = max(len(period_asc) * 3, 60)
    theta_fine_1d = np.linspace(0, 2*np.pi, n_theta_fine, endpoint=False)
    r_fine_1d = np.linspace(period_asc.min(), period_asc.max(), n_r_fine)

    Eplot_fine = spl(r_fine_1d, theta_fine_1d)
    Eplot_fine = np.clip(Eplot_fine, 0, None)

    # Smooth with periodic wrap in angular (column) direction to avoid any seam
    Eplot_fine = gaussian_filter(Eplot_fine, sigma=0.8, mode=('nearest', 'wrap'))

    # Clip to [_vmin, _vmax] so spline/filter overshoot doesn't produce white regions
    Eplot_fine = np.clip(Eplot_fine, _vmin, _vmax)

    # Close the circle for polar plotting by repeating the first column
    theta_fine = np.append(theta_fine_1d, 2*np.pi)
    Eplot_fine = np.hstack([Eplot_fine, Eplot_fine[:, 0:1]])
    theta_fine_grid, r_fine_grid = np.meshgrid(theta_fine, r_fine_1d)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Make grid visible but subtle
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Create custom colormap for wave energy
    cmap = create_wave_energy_colormap()

    # Use filled contours for smoother appearance
    cf = ax.contourf(theta_fine_grid, r_fine_grid, Eplot_fine, levels=levels, cmap=cmap, extend='max', vmin=_vmin, vmax=_vmax)
    # Exclude first/last levels to avoid contour traced along clipped plateau edges
    ax.contour(theta_fine_grid, r_fine_grid, Eplot_fine, levels=levels[1:-1:5], colors='black',
               linewidths=0.3, alpha=0.3)

    # Estilo dos axes
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([5, 10, 15, 20])
    ax.set_yticklabels(['5s', '10s', '15s', '20s'], color='gray', fontsize=7.5)
    ax.set_rlim(0, 25)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='y', colors='gray', labelsize=16)
    ticks = ['N','NE','E','SE','S','SW','W','NW']
    tick_angles = np.deg2rad(np.linspace(0, 315, 8))
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(ticks)
    ax.tick_params(axis='x', colors='k', labelsize=16)
    title = 'Directional Spectrum'
    ax.set_title(title, fontsize=16, color='k', pad=30)

    # Statistics / Wave Systems box
    use_partitions = partitions is not None and len(partitions) > 0
    show_stats = (selected_time is not None or hs is not None or
                  tp is not None or dp is not None or use_partitions)

    if show_stats:
        has_date = selected_time is not None
        if use_partitions:
            # lines: title + date? + (header + Hs + Tp + Dp) per system
            n_lines = 1 + (1 if has_date else 0) + 4 * len(partitions)
        else:
            n_lines = 1 + sum(1 for v in [selected_time, hs, tp, dp] if v is not None)

        line_h = 0.038
        box_h = max(n_lines * line_h + 0.03, 0.15)
        box_left = 0.76
        box_top = 0.92

        stats_ax = fig.add_axes([box_left, box_top - box_h, 0.20, box_h], facecolor='white')
        stats_ax.patch.set_alpha(0.85)
        stats_ax.patch.set_edgecolor('black')
        stats_ax.patch.set_linewidth(1.5)
        stats_ax.axis('off')

        trans = stats_ax.transAxes
        dy = 1.0 / (n_lines + 0.5)
        line_idx = 0

        box_title = 'Wave Systems' if use_partitions else 'Statistics'
        stats_ax.text(0.5, 1.0 - dy * (line_idx + 0.5), box_title,
                      fontsize=13, weight='bold', ha='center', va='top', transform=trans)
        line_idx += 1

        if has_date:
            date_str = selected_time.strftime('%Y-%m-%d %H:%M')
            stats_ax.text(0.5, 1.0 - dy * (line_idx + 0.1), date_str,
                          fontsize=10, ha='center', va='top', transform=trans, color='gray')
            line_idx += 1
            if use_partitions:
                for i, p in enumerate(partitions):
                    stats_ax.text(0.35, 1.0 - dy * (line_idx + 0.15), f'System {i + 1}',
                      fontsize=11, weight='bold', ha='left', va='top',
                      transform=trans, color='navy')
                    line_idx += 0.8
                    stats_ax.text(0.35, 1.0 - dy * (line_idx + 0.05), f"Hs: {p['Hs']:.2f} m",
                          fontsize=10, ha='left', va='top', transform=trans)
                    line_idx += 0.6
                    stats_ax.text(0.35, 1.0 - dy * (line_idx + 0.05), f"Tp: {p['Tp']:.1f} s",
                          fontsize=10, ha='left', va='top', transform=trans)
                    line_idx += 0.6
                    stats_ax.text(0.35, 1.0 - dy * (line_idx + 0.05), f"Dp: {p['Dp']:.0f}\u00b0",
                          fontsize=10, ha='left', va='top', transform=trans)
                    line_idx += 0.8
            else:
                for label, val, fmt in [('Hs', hs, '{:.2f} m'), ('Tp', tp, '{:.1f} s'), ('Dp', dp, '{:.0f}°')]:
                   if val is not None:
                       stats_ax.text(0.35, 1.0 - dy * (line_idx + 0.05), f'{label}: {fmt.format(val)}',
                         fontsize=11, ha='left', va='top', transform=trans)
                line_idx += 0.6

    # Horizontal colorbar at the bottom
    colorbar_label = 'm²·s·rad⁻¹'
    norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.06, 0.63, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_ticks(np.linspace(_vmin, _vmax, 6))

    fig.subplots_adjust(left=0.06, right=0.75, top=0.92, bottom=0.13)

    return fig, ax

