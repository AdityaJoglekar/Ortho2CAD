Orthographic drawing generation

Convert STEP files into orthographic SVG drawings, then convert those SVGs to PNG images.

Usage

1) STEP → SVG

Run the STEP-to-SVG converter. `--input` is a directory of STEP files; `--output` is the directory where SVG files will be written.

```bash
python pythonocc_for_step_to_ortho.py --input /path/to/step/files --output /path/to/output
```

2) SVG → PNG

Convert generated SVGs to PNG images. `--src` is the directory containing SVG files; `--dst` is the output directory for PNGs.

```bash
python svg_to_png.py --src /path/to/svgs --dst /path/to/output
```