#!/usr/bin/env python3
"""
Orthographic projection generator from STEP files using pythonocc-core.
"""

import os
import sys
import argparse
import multiprocessing

# pythonocc-core imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax3, gp_Trsf
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector


class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass


def _process_step_file_worker(step_file_path, output_file_path, verbose=False):
    """Worker function to process a single STEP file in a separate process."""
    try:
        if verbose:
            print(f"Processing: {step_file_path}")
        
        # Create projector
        projector = OrthographicProjector(step_file_path)
        
        # Generate standard views
        views = projector.generate_standard_views()
        
        # Create SVG exporter
        exporter = SVGExporter(width=600, height=600)
        
        # Calculate actual dimensions from the 3D geometry
        actual_dimensions, _ = exporter._calculate_geometry_dimensions(projector.shape)
        if verbose:
            print(f"  Calculated dimensions: width={actual_dimensions['width']:.2f}, height={actual_dimensions['height']:.2f}, depth={actual_dimensions['depth']:.2f}")
        
        # Export combined view with dimensions
        if verbose:
            print(f"  Exporting to: {output_file_path}")
        exporter.export_combined_views(views, output_file_path, include_dimensions=True, actual_dimensions=actual_dimensions)

        return True, None
        
    except Exception as e:
        return False, str(e)


def process_with_timeout(step_file_path, output_file_path, timeout_seconds=10, verbose=False):
    """Process a STEP file with a hard timeout using multiprocessing."""
    try:
        # Create a process for the worker
        process = multiprocessing.Process(
            target=_process_step_file_worker,
            args=(step_file_path, output_file_path, verbose)
        )
        
        # Start the process
        process.start()
        
        # Wait for the process to complete or timeout
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # Process is still running, terminate it
            print(f"  Process timeout after {timeout_seconds}s, terminating...")
            process.terminate()
            process.join(timeout=5)  # Give it 5 seconds to terminate gracefully
            
            if process.is_alive():
                # Force kill if it didn't terminate gracefully
                print(f"  Force killing process...")
                process.kill()
                process.join()
            
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Check if the process completed successfully
        if process.exitcode == 0:
            return True
        else:
            return False
            
    except TimeoutError:
        raise
    except Exception as e:
        print(f"  Error in timeout handler: {e}")
        return False


class OrthographicProjector:
    """Generate orthographic projections with hidden line removal."""
    
    def __init__(self, step_file_path):
        """Initialize with STEP file."""
        self.shape = self._load_step_file(step_file_path)
        if not self.shape:
            raise ValueError(f"Failed to load STEP file: {step_file_path}")
    
    def _load_step_file(self, file_path):
        """Load STEP file and return the shape."""
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return None
        
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file_path)
        
        if status != IFSelect_RetDone:
            print(f"Error: Failed to read STEP file {file_path}")
            return None
        
        # Transfer the shape
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        
        if shape.IsNull():
            print("Error: No valid shape found in STEP file")
            return None
        
        print(f"Successfully loaded STEP file: {file_path}")
        return shape
    
    def make_coordinate_system(self, origin, view_dir, up_dir):
        # Convert gp_Dir to gp_Vec for cross product
        view_vec = gp_Vec(view_dir.X(), view_dir.Y(), view_dir.Z())
        up_vec = gp_Vec(up_dir.X(), up_dir.Y(), up_dir.Z())
        
        # Compute the X direction: perpendicular to Up and View
        x_vec = up_vec.Crossed(view_vec)
        x_vec.Normalize()
        
        # Recompute Up to be perpendicular: View x X
        up_vec = view_vec.Crossed(x_vec)
        up_vec.Normalize()
        
        # Convert back to gp_Dir
        x_dir = gp_Dir(x_vec)
        # Construct Ax3 with corrected X direction
        coordinate_system = gp_Ax3(origin, view_dir, x_dir)
        return coordinate_system


    def create_projector(self, view_direction):
        """Create HLR projector for given view direction using proper technical drawing setup."""
        # Create coordinate system for projection
        origin = gp_Pnt(1000, 1000, 1000)
        

        # Assign up directions for each view (to set local "vertical" axis)
        def get_up(view_direction):
            if abs(view_direction.X()) > 0.5:       # Right view
                return gp_Dir(0, -1, 0)              # Z is up
            elif abs(view_direction.Y()) > 0.5:     # Front view
                return gp_Dir(0, 0, -1)              # Z is up
            else:                                   # Top view
                return gp_Dir(0, 1, 0)              # Y is up


        up_direction = get_up(view_direction)
        # pass up_dir to projection method



        # Create coordinate system with corrected directions
        # coordinate_system = gp_Ax3(origin, view_direction, up_direction)
        coordinate_system = self.make_coordinate_system(origin, view_direction, up_direction)

        
        # Create transformation for the projection
        transformation = gp_Trsf()
        transformation.SetTransformation(coordinate_system, gp_Ax3())
        
        # Initialize the HLR projector with orthographic projection
        # Parameters: transformation, perspective (False for orthographic), focus
        projector = HLRAlgo_Projector(transformation, False, 0.0)
        
        return projector
    
    def generate_hlr_edges(self, projector):
        """Generate hidden line removal edges."""
        try:
            # Set up the HLR algorithm
            hlr_algo = HLRBRep_Algo()
            hlr_algo.Add(self.shape)  # Add the shape
            hlr_algo.Projector(projector)  # Set the projector
            hlr_algo.Update()  # Update the algorithm
            hlr_algo.Hide()  # Perform hidden line removal
            
            # Extract the visible and hidden edges
            hlr_to_shape = HLRBRep_HLRToShape(hlr_algo)
            
            # Extract all types of visible and hidden edges
            visible_edges = []
            hidden_edges = []
            
            # Get main visible and hidden compounds
            try:
                visible_compound = hlr_to_shape.VCompound()
                if not visible_compound.IsNull():
                    visible_edges.append(visible_compound)
            except:
                pass
            
            try:
                hidden_compound = hlr_to_shape.HCompound()
                if not hidden_compound.IsNull():
                    hidden_edges.append(hidden_compound)
            except:
                pass
            
            # Get outline edges (important for external boundaries)
            try:
                visible_outline = hlr_to_shape.OutLineVCompound()
                if not visible_outline.IsNull():
                    visible_edges.append(visible_outline)
            except:
                pass
            
            try:
                hidden_outline = hlr_to_shape.OutLineHCompound()
                if not hidden_outline.IsNull():
                    hidden_edges.append(hidden_outline)
            except:
                pass
            
            # Get smooth and sharp edges separately for better control
            try:
                visible_smooth = hlr_to_shape.Rg1LineVCompound()
                if not visible_smooth.IsNull():
                    visible_edges.append(visible_smooth)
            except:
                pass
            
            try:
                hidden_smooth = hlr_to_shape.Rg1LineHCompound()
                if not hidden_smooth.IsNull():
                    hidden_edges.append(hidden_smooth)
            except:
                pass
            
            print(f"  HLR processed: {len(visible_edges)} visible compounds, {len(hidden_edges)} hidden compounds")
            
            return {
                'visible': visible_edges,
                'hidden': hidden_edges
            }
            
        except Exception as e:
            print(f"HLR processing error: {e}")
            import traceback
            traceback.print_exc()
            return {'visible': [], 'hidden': []}
    
    def edges_to_2d_points(self, compound_shape):
        """Extract 2D points from edge compound."""
        edges_2d = []
        
        if compound_shape.IsNull():
            return edges_2d
        
        explorer = TopExp_Explorer(compound_shape, TopAbs_EDGE)
        edge_count = 0
        
        while explorer.More():
            edge_count += 1
            if edge_count > 1000:  # Safety limit
                print(f"  Warning: Processing stopped at {edge_count} edges to prevent hanging")
                break
                
            edge = explorer.Current()
            
            try:
                # Use BRepAdaptor for more reliable curve access
                adaptor = BRepAdaptor_Curve(edge)
                
                # Get curve type and handle different geometries
                curve_type = adaptor.GetType()
                first_param = adaptor.FirstParameter()
                last_param = adaptor.LastParameter()
                
                # Validate parameter range
                if abs(last_param - first_param) > 1e-12:
                    # Determine number of sample points based on curve type
                    if curve_type == 0:  # Line
                        num_points = 1  # Only need endpoints for lines
                    elif curve_type == 1:  # Circle
                        num_points = 100  # More points for circles
                    elif curve_type == 2:  # Ellipse
                        num_points = 100  # More points for ellipses
                    else:  # Other curves (splines, etc.)
                        num_points = 100  # Moderate sampling
                    
                    points = []
                    
                    for i in range(num_points + 1):
                        try:
                            param = first_param + (last_param - first_param) * i / num_points
                            point = adaptor.Value(param)
                            # Project to 2D (X, Y coordinates) - ensure float conversion
                            points.append((float(point.X()), float(point.Y())))
                        except:
                            # Skip problematic parameter values
                            continue
                    
                    if len(points) >= 2:  # Need at least 2 points for a line
                        edges_2d.append(points)
                            
            except Exception as e:
                # Skip problematic edges silently
                pass
            
            explorer.Next()
        
        print(f"    Extracted {len(edges_2d)} edge curves from {edge_count} edges")
        return edges_2d
    
    def generate_orthographic_view(self, view_direction, view_name="view"):
        """Generate a complete orthographic view with visible and hidden lines."""
        print(f"Generating {view_name} view...")
        
        # Create projector
        print(f"  Creating projector for direction {view_direction.X():.1f}, {view_direction.Y():.1f}, {view_direction.Z():.1f}")
        projector = self.create_projector(view_direction)
        
        # Generate HLR edges
        print(f"  Running HLR algorithm...")
        hlr_result = self.generate_hlr_edges(projector)
        
        # Convert to 2D point lists
        print(f"  Converting visible edges to 2D...")
        visible_edges = []
        for i, compound in enumerate(hlr_result['visible']):
            print(f"    Processing visible compound {i+1}/{len(hlr_result['visible'])}")
            visible_edges.extend(self.edges_to_2d_points(compound))
        
        print(f"  Converting hidden edges to 2D...")
        hidden_edges = []
        for i, compound in enumerate(hlr_result['hidden']):
            print(f"    Processing hidden compound {i+1}/{len(hlr_result['hidden'])}")
            hidden_edges.extend(self.edges_to_2d_points(compound))
        
        print(f"  ✓ {view_name} view completed: {len(visible_edges)} visible, {len(hidden_edges)} hidden edge groups")
        
        return {
            'visible': visible_edges,
            'hidden': hidden_edges,
            'view_name': view_name
        }
    
    def generate_standard_views(self):
        """Generate standard orthographic views (front, top, right) using proper technical drawing conventions."""
        views = {}

        # First angle projection
        # Each direction is the *viewing direction*, i.e. where the camera points (normal to view plane)
        # So, "front" looks from +Y toward -Y
        front_dir = gp_Dir(0, 1, 0)
        views['front'] = self.generate_orthographic_view(front_dir, "Front")

        # "top" looks from +Z toward -Z
        top_dir = gp_Dir(0, 0, 1)
        views['top'] = self.generate_orthographic_view(top_dir, "Top")

        # "right" looks from +X toward -X
        right_dir = gp_Dir(1, 0, 0)
        views['right'] = self.generate_orthographic_view(right_dir, "Right")
        
        return views


class SVGExporter:
    """Export orthographic views to SVG with proper line styling."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.margin = 50
    
    def _points_to_svg_path(self, points):
        """Convert list of points to SVG path string."""
        if len(points) < 2:
            return ""
        
        path = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        for x, y in points[1:]:
            path += f" L {x:.2f},{y:.2f}"
        
        return path
    
    def _calculate_global_scale(self, views_data, view_width, view_height):
        """Calculate a single scale factor to be used across all views for technical drawing accuracy."""
        max_extent = 0
        
        # Find the maximum extent across all views
        for view_name, view_data in views_data.items():
            all_edges = view_data['visible'] + view_data['hidden']
            if not all_edges:
                continue
                
            # Find bounding box for this view
            all_points = []
            for edge in all_edges:
                all_points.extend(edge)
            
            if not all_points:
                continue
                
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            data_width = max_x - min_x
            data_height = max_y - min_y
            
            # Track the maximum extent
            max_extent = max(max_extent, data_width, data_height)
        
        # Calculate global scale factor
        if max_extent > 0:
            scale_x = view_width / max_extent
            scale_y = view_height / max_extent
            global_scale = min(scale_x, scale_y) * 0.9  # Larger scale for better visibility
        else:
            global_scale = 1.0
            
        return global_scale

    def _apply_global_scale_and_center(self, edges, global_scale, view_width, view_height):
        """Apply global scale factor and center the edges in the view area."""
        if not edges:
            return []
            
        # Find bounding box
        all_points = []
        for edge in edges:
            all_points.extend(edge)
        
        if not all_points:
            return []
        
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # Calculate centering offset
        center_x = view_width / 2
        center_y = view_height / 2
        data_center_x = (min_x + max_x) / 2
        data_center_y = (min_y + max_y) / 2
        
        # Transform edges with global scale
        scaled_edges = []
        for edge in edges:
            scaled_edge = []
            for x, y in edge:
                # Apply global scale and center
                new_x = center_x + (x - data_center_x) * global_scale
                new_y = center_y - (y - data_center_y) * global_scale  # Flip Y for SVG
                scaled_edge.append((new_x, new_y))
            scaled_edges.append(scaled_edge)
        
        return scaled_edges
    
    def _rotate_view_90_degrees(self, edges, view_width, view_height):
        """Rotate a view 90 degrees clockwise around the center."""
        if not edges:
            return []
        
        center_x = view_width / 2
        center_y = view_height / 2
        
        rotated_edges = []
        for edge in edges:
            rotated_edge = []
            for x, y in edge:
                # Translate to origin
                tx = x - center_x
                ty = y - center_y
                
                # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
                new_x = ty + center_x
                new_y = -tx + center_y
                
                rotated_edge.append((new_x, new_y))
            rotated_edges.append(rotated_edge)
        
        return rotated_edges
    
    def _calculate_geometry_dimensions(self, shape):
        """Calculate the actual dimensions of the geometry."""
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        dimensions = {
            'width': abs(xmax - xmin),    # X dimension
            'height': abs(zmax - zmin),   # Z dimension (vertical)
            'depth': abs(ymax - ymin)     # Y dimension (depth)
        }
        
        return dimensions, (xmin, ymin, zmin, xmax, ymax, zmax)
    
    def _find_key_dimensions(self, view_name, scaled_visible, scaled_hidden, actual_dimensions):
        """Find the largest overall dimensions for *each view*.

        Requirement:
        - Add the largest dimensions for each view to the final orthographic drawing.
        - Dimension lines should not coincide with other lines: we will *choose offsets later*
          with collision checks, so this method only returns the intended dimension geometry.
        """

        dims = []

        all_points = []
        for edge in scaled_visible:
            all_points.extend(edge)
        if not all_points:
            return dims

        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)

        # Largest horizontal and vertical extents *in this view*.
        view_width = max_x - min_x
        view_height = max_y - min_y

        # Always label with true 3D dimensions when available, else fallback to view extents.
        if actual_dimensions:
            front_like = view_name.lower() == 'front'
            top_like = view_name.lower() == 'top'
            right_like = view_name.lower() == 'right'

            if front_like:
                label_h = f"{actual_dimensions['width']:.4f}"     ##### Changed to 4 decimal places
                label_v = f"{actual_dimensions['height']:.4f}"    ##### Changed to 4 decimal places
            elif top_like:
                label_h = f"{actual_dimensions['width']:.4f}"     ##### Changed to 4 decimal places
                label_v = f"{actual_dimensions['depth']:.4f}"     ##### Changed to 4 decimal places
            elif right_like:
                label_h = f"{actual_dimensions['depth']:.4f}"     ##### Changed to 4 decimal places
                label_v = f"{actual_dimensions['height']:.4f}"    ##### Changed to 4 decimal places
            else:
                label_h = f"{view_width:.4f}"     ##### Changed to 4 decimal places
                label_v = f"{view_height:.4f}"    ##### Changed to 4 decimal places
        else:
            label_h = f"{view_width:.4f}"    ##### Changed to 4 decimal places
            label_v = f"{view_height:.4f}"   ##### Changed to 4 decimal places
        view_key = view_name.lower()

        # Drawing rules:
        # - Remove horizontal dims for Front and Right
        # - Remove vertical dim for Right
        # - Keep both dims for Top
        # => Front: only vertical
        # => Top: horizontal + vertical
        # => Right: none

        if view_key == 'top':
            dims.append({
                'type': 'linear',
                'direction': 'horizontal',
                'start': (min_x, max_y),
                'end': (max_x, max_y),
                'label': label_h,
                'offset_candidates': [-18, -26, -34, -42, 18, 26, 34, 42],
                'priority': 1,
            })

            dims.append({
                'type': 'linear',
                'direction': 'vertical',
                'start': (max_x, min_y),
                'end': (max_x, max_y),
                'label': label_v,
                'offset_candidates': [18, 26, 34, 42, -18, -26, -34, -42],
                'priority': 1,
            })

        elif view_key == 'front':
            dims.append({
                'type': 'linear',
                'direction': 'vertical',
                'start': (max_x, min_y),
                'end': (max_x, max_y),
                'label': label_v,
                'offset_candidates': [18, 26, 34, 42, -18, -26, -34, -42],
                'priority': 1,
            })

        elif view_key == 'right':
            # No dimensions in right view
            pass
        else:
            # Default: keep both
            dims.append({
                'type': 'linear',
                'direction': 'horizontal',
                'start': (min_x, max_y),
                'end': (max_x, max_y),
                'label': label_h,
                'offset_candidates': [-18, -26, -34, -42, 18, 26, 34, 42],
                'priority': 1,
            })
            dims.append({
                'type': 'linear',
                'direction': 'vertical',
                'start': (max_x, min_y),
                'end': (max_x, max_y),
                'label': label_v,
                'offset_candidates': [18, 26, 34, 42, -18, -26, -34, -42],
                'priority': 1,
            })

        print(f"    Proposing {len(dims)} overall dimensions in {view_name} view")
        return dims

    def _collect_segments(self, edges, round_to=2):
        """Convert polyline edges to a set of straight segments for collision tests."""
        segments = []
        for edge in edges:
            if len(edge) < 2:
                continue
            for i in range(len(edge) - 1):
                x1, y1 = edge[i]
                x2, y2 = edge[i + 1]
                segments.append((round(x1, round_to), round(y1, round_to), round(x2, round_to), round(y2, round_to)))
        return segments

    def _segments_bbox(self, segments):
        if not segments:
            return None
        xs = [s[0] for s in segments] + [s[2] for s in segments]
        ys = [s[1] for s in segments] + [s[3] for s in segments]
        return (min(xs), min(ys), max(xs), max(ys))

    def _intersects_segment_bbox(self, bbox, seg):
        x1, y1, x2, y2 = seg
        min_x, min_y, max_x, max_y = bbox
        return not (max(x1, x2) < min_x or min(x1, x2) > max_x or max(y1, y2) < min_y or min(y1, y2) > max_y)

    def _segments_intersect(self, a, b, eps=1e-6):
        """Check if two segments intersect (including touching)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        def orient(x1, y1, x2, y2, x3, y3):
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        def on_seg(x1, y1, x2, y2, x3, y3):
            return min(x1, x2) - eps <= x3 <= max(x1, x2) + eps and min(y1, y2) - eps <= y3 <= max(y1, y2) + eps

        o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

        if (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps):
            if (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps):
                return True

        # Collinear/touching cases
        if abs(o1) <= eps and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if abs(o2) <= eps and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if abs(o3) <= eps and on_seg(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if abs(o4) <= eps and on_seg(bx1, by1, bx2, by2, ax2, ay2):
            return True

        return False

    def _pick_non_overlapping_offset(self, dim, geometry_edges, already_used_segments=None):
        """Pick an offset from candidates so the dimension line avoids coinciding/intersecting geometry."""
        if already_used_segments is None:
            already_used_segments = []

        geom_segments = self._collect_segments(geometry_edges, round_to=2)
        geom_bbox = self._segments_bbox(geom_segments)

        start_x, start_y = dim['start']
        end_x, end_y = dim['end']
        is_horizontal = abs(end_x - start_x) >= abs(end_y - start_y)

        candidates = dim.get('offset_candidates')
        if not candidates:
            candidates = [dim.get('offset', 25)]

        for offset in candidates:
            if is_horizontal:
                y = start_y - offset
                dim_seg = (round(start_x, 2), round(y, 2), round(end_x, 2), round(y, 2))
            else:
                x = start_x + offset
                dim_seg = (round(x, 2), round(start_y, 2), round(x, 2), round(end_y, 2))

            # First, cull early if bbox doesn't overlap at all.
            if geom_bbox and self._intersects_segment_bbox(geom_bbox, dim_seg):
                intersects = False
                for gseg in geom_segments:
                    if self._segments_intersect(dim_seg, gseg):
                        intersects = True
                        break
                if intersects:
                    continue

            # Also avoid overlapping previously drawn dimension lines.
            conflict = False
            for used in already_used_segments:
                if self._segments_intersect(dim_seg, used):
                    conflict = True
                    break
            if conflict:
                continue

            return offset, dim_seg

        # Fallback: last candidate.
        fallback = candidates[-1]
        if is_horizontal:
            y = start_y - fallback
            dim_seg = (round(start_x, 2), round(y, 2), round(end_x, 2), round(y, 2))
        else:
            x = start_x + fallback
            dim_seg = (round(x, 2), round(start_y, 2), round(x, 2), round(end_y, 2))
        return fallback, dim_seg
    
    def _draw_dimension_line(self, x1, y1, x2, y2, dimension_text, offset=20, text_size=10):
        """Draw a dimension line with extension lines and text."""
        # Determine if this is horizontal or vertical dimension
        is_horizontal = abs(x2 - x1) > abs(y2 - y1)
        
        if is_horizontal:
            # Horizontal dimension
            ext_y1 = y1 - offset
            ext_y2 = y2 - offset
            dim_y = min(ext_y1, ext_y2)
            
            # Extension lines
            ext_line1 = f'<line x1="{x1}" y1="{y1}" x2="{x1}" y2="{dim_y - 5}" class="dimension-line"/>'
            ext_line2 = f'<line x1="{x2}" y1="{y2}" x2="{x2}" y2="{dim_y - 5}" class="dimension-line"/>'
            
            # Dimension line
            dim_line = f'<line x1="{x1}" y1="{dim_y}" x2="{x2}" y2="{dim_y}" class="dimension-line"/>'
            
            # Arrowheads
            arrow_size = 3
            arrow1 = f'<polygon points="{x1},{dim_y} {x1+arrow_size},{dim_y-arrow_size/2} {x1+arrow_size},{dim_y+arrow_size/2}" class="dimension-arrow"/>'
            arrow2 = f'<polygon points="{x2},{dim_y} {x2-arrow_size},{dim_y-arrow_size/2} {x2-arrow_size},{dim_y+arrow_size/2}" class="dimension-arrow"/>'
            
            # Text
            text_x = (x1 + x2) / 2
            # text_y = dim_y - 8
            text_y = dim_y + 12    ####Changed from -8 to +12 for 4 decimal places
            text = f'<text x="{text_x}" y="{text_y}" class="dimension-text dimension-text-spaced">{dimension_text}</text>'
            
        else:
            # Vertical dimension
            ext_x1 = x1 + offset
            ext_x2 = x2 + offset
            dim_x = max(ext_x1, ext_x2)
            
            # Extension lines
            ext_line1 = f'<line x1="{x1}" y1="{y1}" x2="{dim_x + 5}" y2="{y1}" class="dimension-line"/>'
            ext_line2 = f'<line x1="{x2}" y1="{y2}" x2="{dim_x + 5}" y2="{y2}" class="dimension-line"/>'
            
            # Dimension line
            dim_line = f'<line x1="{dim_x}" y1="{y1}" x2="{dim_x}" y2="{y2}" class="dimension-line"/>'
            
            # Arrowheads
            arrow_size = 3
            arrow1 = f'<polygon points="{dim_x},{y1} {dim_x-arrow_size/2},{y1+arrow_size} {dim_x+arrow_size/2},{y1+arrow_size}" class="dimension-arrow"/>'
            arrow2 = f'<polygon points="{dim_x},{y2} {dim_x-arrow_size/2},{y2-arrow_size} {dim_x+arrow_size/2},{y2-arrow_size}" class="dimension-arrow"/>'
            
            # Text (rotated for vertical dimensions)
            # SVG text rendering can visually "collapse" the decimal point when rotated.
            # Fix by using a numeric-friendly font and preserving spacing.
            text_x = dim_x + 30   ####Changed from 18 to 30 for 4 decimal places
            text_y = (y1 + y2) / 2
            # text = (
                # f'<text x="{text_x}" y="{text_y}" class="dimension-text dimension-text-spaced" '
                # f'text-rendering="geometricPrecision" '
                # f'transform="rotate(-90 {text_x} {text_y})">{dimension_text}</text>'
            # )
            text = f'<text x="{text_x}" y="{text_y}" class="dimension-text dimension-text-spaced">{dimension_text}</text>'
        
        return f'{ext_line1}\n        {ext_line2}\n        {dim_line}\n        {arrow1}\n        {arrow2}\n        {text}'
    
    def _add_view_dimensions(self, view_name, scaled_visible, scaled_hidden, view_width, view_height, actual_dimensions):
        """Add intelligent dimensions to a specific view with feature detection and redundancy reduction."""
        dimensions_svg = ""
        
        if not scaled_visible:
            return dimensions_svg
        
        # Find key dimensions for this view
        key_dimensions = self._find_key_dimensions(view_name, scaled_visible, scaled_hidden, actual_dimensions)
        
        print(f"  Found {len(key_dimensions)} dimensions for {view_name} view")
        
        # Sort by priority (lower number = higher priority)
        key_dimensions.sort(key=lambda d: d.get('priority', 5))
        
        # Draw each dimension, choosing offsets that avoid overlapping geometry.
        used_dim_segments = []
        geometry_edges_for_collision = list(scaled_visible) + list(scaled_hidden)
        for dim in key_dimensions:
            try:
                if dim['type'] == 'linear':
                    start_x, start_y = dim['start']
                    end_x, end_y = dim['end']
                    label = dim['label']
                    offset, dim_seg = self._pick_non_overlapping_offset(
                        dim,
                        geometry_edges_for_collision,
                        already_used_segments=used_dim_segments,
                    )
                    used_dim_segments.append(dim_seg)
                    
                    dim_svg = self._draw_dimension_line(start_x, start_y, end_x, end_y, label, offset)
                    dimensions_svg += f"        {dim_svg}\n"
                    
            except Exception as e:
                print(f"  Warning: Failed to draw dimension {dim.get('type', 'unknown')}: {e}")
                continue
        
        return dimensions_svg
    
    def export_combined_views(self, views_data, filename, include_dimensions=True, actual_dimensions=None):
        """Export multiple views to a single SVG with uniform scaling across all views."""
        view_width = 300  # Increased from calculated value for bigger figures
        view_height = 300  # Increased from calculated value for bigger figures
        
        # Use provided dimensions or try to calculate from shape
        if include_dimensions and actual_dimensions:
            pass  # Use provided dimensions
        elif include_dimensions and hasattr(self, 'shape'):
            actual_dimensions, _ = self._calculate_geometry_dimensions(self.shape)
        else:
            actual_dimensions = None
        
        # Calculate global scale factor for all views
        global_scale = self._calculate_global_scale(views_data, view_width, view_height)
        print(f"Using global scale factor: {global_scale:.4f}")
        
        # Canvas size for larger figures
        canvas_width = 800
        canvas_height = 800
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{canvas_width}" height="{canvas_height}" xmlns="http://www.w3.org/2000/svg">
    <title>Orthographic Views</title>
    
    <!-- Define line styles -->
    <defs>
        <style>
            .visible-line {{
                stroke: #000000;
                stroke-width: 1.5;
                fill: none;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
            .hidden-line {{
                stroke: #000000;
                stroke-width: 1.5;
                fill: none;
                stroke-dasharray: 1,3;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
            .view-title {{
                font-family: Arial, sans-serif;
                font-size: 14;
                font-weight: bold;
                text-anchor: middle;
            }}
            .dimension-line {{
                stroke: #000000;
                stroke-width: 0.8;
                fill: none;
            }}
            .dimension-arrow {{
                stroke: #000000;
                stroke-width: 0.8;
                fill: #000000;
            }}
            .dimension-text {{
                font-family: Arial, sans-serif;
                font-size: 14;
                fill: #000000;
                text-anchor: middle;
                dominant-baseline: middle;
            }}
            .dimension-text-spaced {{
                letter-spacing: 0.35px;
            }}
        </style>
    </defs>
    
    <!-- Background -->
    <rect width="{canvas_width}" height="{canvas_height}" fill="white"/>
    
'''
        # <!-- Main title -->
        # <text x="{canvas_width//2}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold">Orthographic Views</text>
        
        # Position views according to technical drawing standards with larger spacing
        # Front view in center, Top view below, Right view to the left
        positions = {
            'front': (400, 100),      # Center position
            'top': (400, 450),        # Below front view  
            'right': (50, 100)        # Left of front view (right side view)
        }
        
        for view_name, view_data in views_data.items():
            if view_name not in positions:
                continue
            
            x_offset, y_offset = positions[view_name]
            
            # Apply global scale to all edges (uniform scaling across all views)
            all_edges = view_data['visible'] + view_data['hidden']
            scaled_all = self._apply_global_scale_and_center(all_edges, global_scale, view_width, view_height)
            
            # Rotate right view 90 degrees clockwise to correct orientation
            if view_name == 'right':
                scaled_all = self._rotate_view_90_degrees(scaled_all, view_width, view_height)
            
            # Split back into visible and hidden after scaling
            num_visible = len(view_data['visible'])
            scaled_visible = scaled_all[:num_visible]
            scaled_hidden_unfiltered = scaled_all[num_visible:]

            scaled_hidden = scaled_hidden_unfiltered
            
            # Add view group
            svg_content += f'\n    <!-- {view_data["view_name"]} View -->\n'
            svg_content += f'    <g transform="translate({x_offset},{y_offset})">\n'
            svg_content += f'        <text x="{view_width/2}" y="-10" class="view-title">{view_data["view_name"]} View</text>\n'
            # svg_content += f'        <rect width="{view_width}" height="{view_height}" fill="none" stroke="#cccccc" stroke-width="0.5"/>\n'
            
            # Add visible edges
            for edge in scaled_visible:
                if len(edge) >= 2:
                    path = self._points_to_svg_path(edge)
                    svg_content += f'        <path d="{path}" class="visible-line"/>\n'
            
            # Add hidden edges
            for edge in scaled_hidden:
                if len(edge) >= 2:
                    path = self._points_to_svg_path(edge)
                    svg_content += f'        <path d="{path}" class="hidden-line"/>\n'
            
            # Add dimensions if enabled
            if include_dimensions and actual_dimensions:
                dimensions_svg = self._add_view_dimensions(
                    view_data["view_name"], 
                    scaled_visible, 
                    scaled_hidden,  # Include hidden edges for feature detection
                    view_width, 
                    view_height, 
                    actual_dimensions
                )
                svg_content += dimensions_svg
            
            svg_content += '    </g>\n'
        
        svg_content += '</svg>'
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(svg_content)
        
        print(f"Exported combined orthographic views to {filename}")


def process_single_step_file(step_file_path, output_file_path, verbose=False, timeout_seconds=10):
    """Process a single STEP file and generate orthographic views."""
    try:
        # Use multiprocessing-based timeout
        return process_with_timeout(step_file_path, output_file_path, timeout_seconds, verbose)
        
    except TimeoutError as e:
        print(f"  Timeout processing {step_file_path}: {e}")
        return False
    except Exception as e:
        print(f"  Error processing {step_file_path}: {e}")
        return False


def find_step_files(directory):
    """Recursively find all STEP files in the given directory."""
    step_files = []
    step_extensions = ['.step', '.stp', '.STEP', '.STP']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in step_extensions):
                step_files.append(os.path.join(root, file))
    
    return step_files


def create_output_structure(steps_dir, output_dir, step_file_path):
    """Create the corresponding output directory structure and return the output file path."""
    # Get relative path from steps directory
    rel_path = os.path.relpath(step_file_path, steps_dir)
    
    # Change extension to .svg
    rel_path_no_ext = os.path.splitext(rel_path)[0]
    output_rel_path = rel_path_no_ext + ".svg"
    
    # Create full output path
    output_file_path = os.path.join(output_dir, output_rel_path)
    
    # Create output directory if it doesn't exist
    output_file_dir = os.path.dirname(output_file_path)
    os.makedirs(output_file_dir, exist_ok=True)
    
    return output_file_path


def main():
    """Main function to process all STEP files in steps folder."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate orthographic projections from STEP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pythonocc_for_step_to_ortho.py
  python pythonocc_for_step_to_ortho.py --input my_steps --output my_svgs
  python pythonocc_for_step_to_ortho.py -i /path/to/step/files -o /path/to/output
  python pythonocc_for_step_to_ortho.py --timeout 30 --verbose
        """
    )
    
    parser.add_argument(
        '-i', '--input', 
        default='steps',
        help='Input directory containing STEP files (default: steps)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='orthographic_out', 
        help='Output directory for SVG files (default: orthographic_out)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=10,
        help='Timeout in seconds for processing each STEP file (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    steps_dir = args.input
    output_dir = args.output
    
    # Check if steps directory exists
    if not os.path.exists(steps_dir):
        print(f"Error: Steps directory '{steps_dir}' does not exist")
        print("Please create a 'steps' folder and place your STEP files in it")
        return 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all STEP files
    print(f"Scanning for STEP files in '{steps_dir}'...")
    step_files = find_step_files(steps_dir)
    
    if not step_files:
        print(f"No STEP files found in '{steps_dir}' directory")
        print("Supported extensions: .step, .stp, .STEP, .STP")
        return 1
    
    print(f"Found {len(step_files)} STEP files to process")
    
    # Process each STEP file
    successful = 0
    failed = 0
    
    for i, step_file in enumerate(step_files, 1):
        print(f"\n[{i}/{len(step_files)}] Processing: {step_file}")
        
        # Create output file path maintaining directory structure
        output_file_path = create_output_structure(steps_dir, output_dir, step_file)

        # Process the file
        if process_single_step_file(step_file, output_file_path, verbose=args.verbose, timeout_seconds=args.timeout):
            successful += 1
            print(f"  ✓ Successfully processed")
        else:
            failed += 1
            print(f"  ✗ Failed to process")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total files processed: {len(step_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\nThe SVG files contain:")
        print("- Solid black lines for visible edges")
        print("- Dashed gray lines for hidden edges")
        print("- Proper orthographic projections with hidden line removal")
        print("- Combined front, top, and right views")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    multiprocessing.freeze_support()
    sys.exit(main())
