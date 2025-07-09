import h5py
import numpy as np
import sleap
from sleap.instance import Point, PredictedPoint
from pathlib import Path

def extract_and_convert_h5_to_sleap(h5_path, video_path, output_slp_path):
    """Extract H5 data and convert directly to SLEAP format"""
    
    with h5py.File(h5_path, 'r') as f:
        # Get the main data
        pose_data = f['df_with_missing/block0_values'][...]  # Shape: (674, 510)
        
        # Get metadata
        h5_bodyparts = [bp.decode('utf-8') for bp in f['df_with_missing/axis0_level2'][...]]
        animals = [animal.decode('utf-8') for animal in f['df_with_missing/axis0_level1'][...]]
        coords = [coord.decode('utf-8') for coord in f['df_with_missing/axis0_level3'][...]]
        
        print(f"H5 bodyparts (in order): {h5_bodyparts}")
        print(f"Animals: {animals}")
        print(f"Coordinates: {coords}")
        print(f"Pose data shape: {pose_data.shape}")
        
        # Define COCO skeleton order (what SLEAP expects)
        coco_bodyparts = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Create mapping from H5 order to COCO order with manual corrections
        h5_to_coco_mapping = []
        for coco_bp in coco_bodyparts:
            # this should be where that is
            h5_bp = coco_bp
            # -----
            if coco_bp == "right_elbow":
                h5_bp = "nose"
            elif coco_bp == "left_ear":
                h5_bp = "left_eye"
            elif coco_bp == "right_hip":
                h5_bp = "right_eye"
            elif coco_bp == "left_eye":
                h5_bp = "left_ear"
            elif coco_bp == "right_wrist":
                h5_bp = "right_ear"
            elif coco_bp == "right_shoulder":
                h5_bp = "left_shoulder"
            elif coco_bp == "left_ankle":
                h5_bp = "right_shoulder"
            elif coco_bp == "right_eye":
                h5_bp = "left_elbow"
            elif coco_bp == "left_hip":
                h5_bp = "right_elbow"
            elif coco_bp == "left_elbow":
                h5_bp = "left_wrist"
            elif coco_bp == "right_ankle":
                h5_bp = "right_wrist"
            elif coco_bp == "right_ear":
                h5_bp = "left_hip"
            elif coco_bp == "left_knee":
                h5_bp = "right_hip"
            elif coco_bp == "left_shoulder":
                h5_bp = "left_knee"
            elif coco_bp == "right_knee":
                h5_bp = "right_knee"
            elif coco_bp == "nose":
                h5_bp = "left_ankle"
            elif coco_bp == "left_wrist":
                h5_bp = "right_ankle"
            # -------
            if h5_bp in h5_bodyparts:
                h5_idx = h5_bodyparts.index(h5_bp)
                h5_to_coco_mapping.append(h5_idx)
                print(f"  {coco_bp} -> H5 '{h5_bp}' (index {h5_idx})")
            else:
                print(f"  WARNING: {h5_bp} not found in H5 data!")
                h5_to_coco_mapping.append(-1)  # Mark as missing
        
        # Reshape data: (frames, animals, bodyparts, coords)
        n_frames = pose_data.shape[0]
        n_animals = len(animals)
        n_bodyparts = len(h5_bodyparts)
        n_coords = len(coords)
        
        reshaped_data = pose_data.reshape(n_frames, n_animals, n_bodyparts, n_coords)
        print(f"Reshaped to: {reshaped_data.shape}")
        
        # Create SLEAP skeleton with proper human pose edges
        skeleton = sleap.Skeleton.from_names_and_edge_inds(
            node_names=coco_bodyparts,
            edge_inds=[
                # Head connections
                (0, 1),   # nose -> left_eye
                (0, 2),   # nose -> right_eye
                (1, 3),   # left_eye -> left_ear
                (2, 4),   # right_eye -> right_ear
                
                # Torso connections
                (5, 6),   # left_shoulder -> right_shoulder
                (5, 11),  # left_shoulder -> left_hip
                (6, 12),  # right_shoulder -> right_hip
                (11, 12), # left_hip -> right_hip
                
                # Left arm connections
                (5, 7),   # left_shoulder -> left_elbow
                (7, 9),   # left_elbow -> left_wrist
                
                # Right arm connections
                (6, 8),   # right_shoulder -> right_elbow
                (8, 10),  # right_elbow -> right_wrist
                
                # Left leg connections
                (11, 13), # left_hip -> left_knee
                (13, 15), # left_knee -> left_ankle
                
                # Right leg connections
                (12, 14), # right_hip -> right_knee
                (14, 16), # right_knee -> right_ankle
            ]
        )
        
        # Load video
        try:
            video = sleap.Video.from_filename(str(video_path))
            print(f"Loaded video: {video_path}")
            print(f"Video shape: {video.shape}")
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
        
        # Create empty labels container - fix initialization
        labeled_frames = []
        
        # Process each frame
        valid_frames = 0
        total_instances = 0
        confidence_threshold = 0.1  # Minimum confidence for valid keypoints
        
        for frame_idx in range(n_frames):
            instances = []
            
            # Process each animal in this frame
            for animal_idx in range(n_animals):
                points = {}
                valid_points = 0
                
                # Process each bodypart in COCO order
                for coco_idx, coco_bp in enumerate(coco_bodyparts):
                    h5_idx = h5_to_coco_mapping[coco_idx]
                    
                    if h5_idx >= 0:  # Valid mapping
                        # Get coordinates - Try different interpretations
                        coord1 = reshaped_data[frame_idx, animal_idx, h5_idx, 0]
                        coord2 = reshaped_data[frame_idx, animal_idx, h5_idx, 1] 
                        coord3 = reshaped_data[frame_idx, animal_idx, h5_idx, 2]
                        
                        # DeepLabCut format is typically [x, y, likelihood]
                        x = coord1
                        y = coord2
                        likelihood = coord3
                        
                        # Convert to pixel coordinates (no transformation needed if already in pixels)
                        x_pixel = float(x)
                        y_pixel = float(y)
                        
                        # Validate coordinates
                        if (x_pixel > 0 and y_pixel > 0 and  # Positive coordinates
                            x_pixel < 1920 and y_pixel < 1080 and  # Within video bounds
                            likelihood > confidence_threshold and
                            not (np.isnan(x) or np.isnan(y) or np.isnan(likelihood))):
                            
                            # Create PredictedPoint with corrected coordinates
                            predicted_point = PredictedPoint(
                                x=x_pixel, 
                                y=y_pixel, 
                                score=float(likelihood)
                            )
                            points[coco_bp] = predicted_point
                            valid_points += 1
                
                # Only create instance if we have enough valid points
                if valid_points >= 5:  # Require at least 5 valid keypoints
                    instance = sleap.Instance(skeleton=skeleton, points=points)
                    instances.append(instance)
                    total_instances += 1
            
            # Create labeled frame if we have instances
            if instances:
                labeled_frame = sleap.LabeledFrame(
                    video=video, 
                    frame_idx=frame_idx, 
                    instances=instances
                )
                labeled_frames.append(labeled_frame)
                valid_frames += 1
                
                if valid_frames % 50 == 0:
                    print(f"Processed {valid_frames} valid frames...")
        
        # Create labels from the labeled frames
        labels = sleap.Labels(labeled_frames)
        
        # Save SLEAP project
        try:
            labels.save(str(output_slp_path))
            print(f"\n=== CONVERSION COMPLETE ===")
            print(f"Valid frames: {valid_frames}/{n_frames}")
            print(f"Total instances: {total_instances}")
            # Fix division by zero error
            if valid_frames > 0:
                print(f"Average instances per frame: {total_instances/valid_frames:.2f}")
            else:
                print("No valid frames found - check data quality and confidence thresholds")
            print(f"Output saved to: {output_slp_path}")
            return labels
        except Exception as e:
            print(f"Error saving SLEAP file: {e}")
            return None

def analyze_h5_data_quality(h5_path):
    """Analyze data quality and distribution"""
    
    with h5py.File(h5_path, 'r') as f:
        pose_data = f['df_with_missing/block0_values'][...]
        bodyparts = [bp.decode('utf-8') for bp in f['df_with_missing/axis0_level2'][...]]
        
        reshaped = pose_data.reshape(pose_data.shape[0], 10, 17, 3)
        
        print("=== DATA QUALITY ANALYSIS ===")
        
        # Overall statistics
        total_keypoints = reshaped.shape[0] * reshaped.shape[1] * reshaped.shape[2]
        valid_keypoints = np.sum((reshaped[:, :, :, 1] > 0) & (reshaped[:, :, :, 2] > 0))
        
        print(f"Total possible keypoints: {total_keypoints}")
        print(f"Valid keypoints (x>0, y>0): {valid_keypoints}")
        print(f"Data completeness: {valid_keypoints/total_keypoints*100:.1f}%")
        
        # Per-frame analysis
        frames_with_data = 0
        for frame_idx in range(min(10, reshaped.shape[0])):
            frame_data = reshaped[frame_idx]
            instances_in_frame = 0
            
            for animal_idx in range(10):
                animal_data = frame_data[animal_idx]
                valid_points = np.sum((animal_data[:, 1] > 0) & (animal_data[:, 2] > 0))
                
                if valid_points >= 5:
                    instances_in_frame += 1
            
            # Only report if frame has valid instances
            if instances_in_frame > 0:
                frames_with_data += 1
                print(f"Frame {frame_idx}: {instances_in_frame} instances")
                
                # Show confidence distribution for first valid instance
                for animal_idx in range(10):
                    animal_data = frame_data[animal_idx]
                    valid_points = np.sum((animal_data[:, 1] > 0) & (animal_data[:, 2] > 0))
                    if valid_points >= 5:
                        # Safe confidence extraction
                        try:
                            valid_mask = (animal_data[:, 1] > 0) & (animal_data[:, 2] > 0)
                            if np.any(valid_mask):
                                confidences = animal_data[valid_mask, 0]
                                if len(confidences) > 0:
                                    print(f"  Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
                        except Exception as e:
                            print(f"  Could not extract confidence values: {e}")
                        break
        
        print(f"Frames with valid data (first 10): {frames_with_data}/10")

def debug_coordinate_format(h5_path, frame_idx=0, animal_idx=0):
    """Debug the actual coordinate format in the H5 file"""
    
    with h5py.File(h5_path, 'r') as f:
        pose_data = f['df_with_missing/block0_values'][...]
        bodyparts = [bp.decode('utf-8') for bp in f['df_with_missing/axis0_level2'][...]]
        animals = [animal.decode('utf-8') for animal in f['df_with_missing/axis0_level1'][...]]
        coords = [coord.decode('utf-8') for coord in f['df_with_missing/axis0_level3'][...]]
        
        print(f"Coordinate labels: {coords}")
        
        # Reshape and examine specific data points
        reshaped = pose_data.reshape(pose_data.shape[0], len(animals), len(bodyparts), len(coords))
        
        print(f"\nExamining frame {frame_idx}, animal {animal_idx}:")
        print("Bodypart -> [coord1, coord2, coord3]")
        
        for bp_idx, bp in enumerate(bodyparts[:5]):  # First 5 bodyparts
            values = reshaped[frame_idx, animal_idx, bp_idx, :]
            print(f"{bp:15} -> [{values[0]:8.3f}, {values[1]:8.3f}, {values[2]:8.3f}]")
        
        # Try to infer format based on value ranges
        print(f"\nValue range analysis:")
        all_coords = reshaped[frame_idx, animal_idx, :, :]
        for i in range(3):
            valid_vals = all_coords[:, i][~np.isnan(all_coords[:, i])]
            if len(valid_vals) > 0:
                print(f"Coord {i}: min={valid_vals.min():.3f}, max={valid_vals.max():.3f}, mean={valid_vals.mean():.3f}")

if __name__ == "__main__":
    h5_file = "./videos/pivot/humanbody_latest/pivot_cut_superanimal_humanbody_rtmpose_x_fasterrcnn_resnet50_fpn_v2.h5"
    video_file = "./videos/pivot/pivot_cut.mp4"
    output_file = "./human_poses.slp"
    
    # Debug coordinate format first
    debug_coordinate_format(h5_file)
    
    # Then analyze data quality
    analyze_h5_data_quality(h5_file)
    
    print("\n" + "="*60)
    
    # Convert to SLEAP
    try:
        labels = extract_and_convert_h5_to_sleap(h5_file, video_file, output_file)
        if labels:
            print(f"\nSUCCESS: Converted to SLEAP format!")
            print(f"You can now open {output_file} in SLEAP GUI")
        else:
            print("\nFAILED: Conversion unsuccessful")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()