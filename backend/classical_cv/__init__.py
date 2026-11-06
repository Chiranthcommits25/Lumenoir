"""Classical computer vision primitives.

Implemented:

- ``arrays``: validation, luma and smoothstep helpers
- ``filters``: box filter and guided filter (§3)
- ``noise``: Immerkaer noise sigma and local noise map (§3.6)
- ``illumination``: LIME-style illumination estimation (§3.1)
- ``colour``: LAB colour-cast detection (§3.7, detection only)
- ``scan``: LumenScan statistics and rule-based parameter selection (§5)
- ``retinex``: LumenRetinex illumination correction, highlight protection and
  CLAHE on LAB L* (§3.3, §3.4, §3.8)
- ``edge_guard``: gain-aware luma guided filter and chroma bilateral filter
  (§3.5)
- ``hue_lock``: guarded LAB cast correction (§3.7)

Not implemented yet: LumenFuse. Importing this package stays light; the
submodules import NumPy/OpenCV only when used.
"""
