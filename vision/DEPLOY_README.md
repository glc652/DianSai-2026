# Raspberry Pi deployment checkpoint

Bundle: `vision_rpi_2026-08-01_mode3_inverse_green_checkpoint`

This bundle freezes the current Raspberry Pi runtime before experimenting
with a different centre-divider colour. It includes the latest target-layout
and Mode3 contour changes, plus the optional C++/OpenMP source.

## Important status

- Native automated suite: `54 passed`.
- Forced Python fallback: `50 passed, 4 skipped`.
- Retained Mode3 image regression: `14/14` output.
- The supplied 7-clubs field image has improved `4,4,3,4` polygon input but
  still produces a safe geometry rejection. It does not send false actions.
- This is a diagnostic checkpoint, not a replacement claim for the last
  competition-stable release.

## Install

```bash
mkdir -p ~/vision
unzip -o ~/vision_rpi_2026-08-01_mode3_inverse_green_checkpoint.zip -d ~/vision
cd ~/vision
chmod +x install_rpi.sh run_rpi.sh
./install_rpi.sh
```

The native extension must be built on the Raspberry Pi. Do not copy a
Windows `.pyd` or another Pi's ABI-specific `.so`.

## Calibration

`rpi_camera_calibration.json` is deliberately absent. The bundled
`rpi_camera_calibration.reference.json` is reference-only. Keep the target
Pi's existing calibration only when camera, lens, mount, resolution and
rotation are unchanged; otherwise perform Mode0 again.

## Run

```bash
cd ~/vision
./run_rpi.sh
```

The launcher uses the old USB-camera profile by default, three solver workers,
the browser preview on port 8080, and `/dev/serial0` at 115200 baud.
