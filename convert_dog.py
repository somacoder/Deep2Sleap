import h5py
import numpy as np
import pandas as pd
import sleap
from sleap.instance import Point, PredictedPoint
from pathlib import Path

def extract_and_convert_dog_h5_to_sleap(h5_path, video_path, output_slp_path):
    """Extract dog H5 data and convert directly to SLEAP format"""
    
    # Read the pandas DataFrame
    df = pd.read_hdf(h5_path)
    
    # Extract structure info
    scorer = df.columns.levels[0][0]
    animals = list(df.columns.levels[1])
    bodyparts = list(df.columns.levels[2])
    coords = list(df.columns.levels[3])  # ['likelihood', 'x', 'y']
    
    print(f"Scorer: {scorer}")
    print(f"Animals: {animals}")
    print(f"Bodyparts ({len(bodyparts)}): {bodyparts}")
    print(f"Coordinates: {coords}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Number of frames: {len(df)}")
    
    # Filter out antler bodyparts since they're not relevant for dogs
    relevant_bodyparts = [bp for bp in bodyparts if 'antler' not in bp.lower()]
    
    print(f"Original bodyparts ({len(bodyparts)}): {bodyparts}")
    print(f"Relevant bodyparts ({len(relevant_bodyparts)}): {relevant_bodyparts}")
    
    # Create SLEAP skeleton with only relevant quadruped bodyparts
    skeleton = sleap.Skeleton.from_names_and_edge_inds(
        node_names=relevant_bodyparts,
        edge_inds=[
            # Face/Head core
            (relevant_bodyparts.index('nose'), relevant_bodyparts.index('upper_jaw')),
            (relevant_bodyparts.index('nose'), relevant_bodyparts.index('lower_jaw')),
            (relevant_bodyparts.index('upper_jaw'), relevant_bodyparts.index('lower_jaw')),
            
            # Eyes to face
            (relevant_bodyparts.index('left_eye'), relevant_bodyparts.index('nose')),
            (relevant_bodyparts.index('right_eye'), relevant_bodyparts.index('nose')),
            
            # Ears (base to end, and base to eyes)
            (relevant_bodyparts.index('left_earbase'), relevant_bodyparts.index('left_earend')),
            (relevant_bodyparts.index('right_earbase'), relevant_bodyparts.index('right_earend')),
            (relevant_bodyparts.index('left_earbase'), relevant_bodyparts.index('left_eye')),
            (relevant_bodyparts.index('right_earbase'), relevant_bodyparts.index('right_eye')),
            
            # Mouth corners to jaw
            (relevant_bodyparts.index('mouth_end_left'), relevant_bodyparts.index('lower_jaw')),
            (relevant_bodyparts.index('mouth_end_right'), relevant_bodyparts.index('lower_jaw')),
            
            # Neck/throat connection
            (relevant_bodyparts.index('neck_end'), relevant_bodyparts.index('throat_end')),
            (relevant_bodyparts.index('neck_base'), relevant_bodyparts.index('throat_base')),
            (relevant_bodyparts.index('neck_end'), relevant_bodyparts.index('upper_jaw')),
            (relevant_bodyparts.index('throat_end'), relevant_bodyparts.index('lower_jaw')),
            
            # Main spine (the core body structure)
            (relevant_bodyparts.index('neck_base'), relevant_bodyparts.index('back_base')),
            (relevant_bodyparts.index('back_base'), relevant_bodyparts.index('back_middle')),
            (relevant_bodyparts.index('back_middle'), relevant_bodyparts.index('back_end')),
            
            # Body width connections
            (relevant_bodyparts.index('body_middle_left'), relevant_bodyparts.index('body_middle_right')),
            (relevant_bodyparts.index('back_middle'), relevant_bodyparts.index('body_middle_left')),
            (relevant_bodyparts.index('back_middle'), relevant_bodyparts.index('body_middle_right')),
            
            # Belly connection
            (relevant_bodyparts.index('belly_bottom'), relevant_bodyparts.index('back_middle')),
            
            # Front legs - connect to front of body
            (relevant_bodyparts.index('back_base'), relevant_bodyparts.index('front_left_thai')),
            (relevant_bodyparts.index('front_left_thai'), relevant_bodyparts.index('front_left_knee')),
            (relevant_bodyparts.index('front_left_knee'), relevant_bodyparts.index('front_left_paw')),
            
            (relevant_bodyparts.index('back_base'), relevant_bodyparts.index('front_right_thai')),
            (relevant_bodyparts.index('front_right_thai'), relevant_bodyparts.index('front_right_knee')),
            (relevant_bodyparts.index('front_right_knee'), relevant_bodyparts.index('front_right_paw')),
            
            # Back legs - connect to back of body
            (relevant_bodyparts.index('back_end'), relevant_bodyparts.index('back_left_thai')),
            (relevant_bodyparts.index('back_left_thai'), relevant_bodyparts.index('back_left_knee')),
            (relevant_bodyparts.index('back_left_knee'), relevant_bodyparts.index('back_left_paw')),
            
            (relevant_bodyparts.index('back_end'), relevant_bodyparts.index('back_right_thai')),
            (relevant_bodyparts.index('back_right_thai'), relevant_bodyparts.index('back_right_knee')),
            (relevant_bodyparts.index('back_right_knee'), relevant_bodyparts.index('back_right_paw')),
            
            # Tail
            (relevant_bodyparts.index('back_end'), relevant_bodyparts.index('tail_base')),
            (relevant_bodyparts.index('tail_base'), relevant_bodyparts.index('tail_end')),
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
    
    # Create empty labels container
    labeled_frames = []
    
    # Process each frame
    valid_frames = 0
    total_instances = 0
    confidence_threshold = 0.1  # Minimum confidence for valid keypoints
    
    for frame_idx in range(len(df)):
        instances = []
        
        # Process each animal in this frame
        for animal in animals:
            points = {}
            valid_points = 0
            
            # Process each bodypart
            for bodypart in relevant_bodyparts:  # Changed from 'bodyparts'
                try:
                    # Get coordinates: (scorer, animal, bodypart, coordinate)
                    likelihood = df.loc[frame_idx, (scorer, animal, bodypart, 'likelihood')]
                    x = df.loc[frame_idx, (scorer, animal, bodypart, 'x')]
                    y = df.loc[frame_idx, (scorer, animal, bodypart, 'y')]
                    
                    # Handle missing data (NaN or negative values)
                    if (pd.notna(x) and pd.notna(y) and pd.notna(likelihood) and
                        x > 0 and y > 0 and likelihood > confidence_threshold):
                        
                        # Convert coordinates to pixel space
                        x_pixel = float(x)
                        y_pixel = float(y)
                        
                        # Validate coordinates are within video bounds
                        if (x_pixel < 1920 and y_pixel < 1080):
                            # Create PredictedPoint
                            predicted_point = PredictedPoint(
                                x=x_pixel, 
                                y=y_pixel, 
                                score=float(likelihood)
                            )
                            points[bodypart] = predicted_point
                            valid_points += 1
                
                except (KeyError, ValueError):
                    # Skip missing or invalid data
                    continue
            
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
        print(f"\n=== DOG CONVERSION COMPLETE ===")
        print(f"Valid frames: {valid_frames}/{len(df)}")
        print(f"Total instances: {total_instances}")
        if valid_frames > 0:
            print(f"Average instances per frame: {total_instances/valid_frames:.2f}")
        else:
            print("No valid frames found - check data quality and confidence thresholds")
        print(f"Output saved to: {output_slp_path}")
        return labels
    except Exception as e:
        print(f"Error saving SLEAP file: {e}")
        return None

def analyze_dog_data_quality(h5_path):
    """Analyze dog data quality"""
    
    df = pd.read_hdf(h5_path)
    scorer = df.columns.levels[0][0]
    animals = list(df.columns.levels[1])
    bodyparts = list(df.columns.levels[2])
    
    print("=== DOG DATA QUALITY ANALYSIS ===")
    
    # Check first few frames
    frames_with_data = 0
    for frame_idx in range(min(10, len(df))):
        instances_in_frame = 0
        
        for animal in animals:
            valid_points = 0
            for bodypart in bodyparts[:5]:  # Check first 5 bodyparts
                try:
                    x = df.loc[frame_idx, (scorer, animal, bodypart, 'x')]
                    y = df.loc[frame_idx, (scorer, animal, bodypart, 'y')]
                    likelihood = df.loc[frame_idx, (scorer, animal, bodypart, 'likelihood')]
                    
                    if pd.notna(x) and pd.notna(y) and x > 0 and y > 0 and likelihood > 0.1:
                        valid_points += 1
                except:
                    continue
            
            if valid_points >= 3:
                instances_in_frame += 1
        
        if instances_in_frame > 0:
            frames_with_data += 1
            print(f"Frame {frame_idx}: {instances_in_frame} instances")
    
    print(f"Frames with valid data (first 10): {frames_with_data}/10")

if __name__ == "__main__":
    h5_file = "./videos/pivot/superanimal_latest/pivot_cut_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32.h5"
    video_file = "./videos/pivot/pivot_cut.mp4"
    output_file = "./dog_poses.slp"
    
    # Check files exist
    if not Path(h5_file).exists():
        print(f"Error: H5 file not found: {h5_file}")
        exit(1)
    
    if not Path(video_file).exists():
        print(f"Error: Video file not found: {video_file}")
        exit(1)
    
    # Analyze data quality
    analyze_dog_data_quality(h5_file)
    
    print("\n" + "="*60)
    
    # Convert to SLEAP
    try:
        labels = extract_and_convert_dog_h5_to_sleap(h5_file, video_file, output_file)
        if labels:
            print(f"\nSUCCESS: Converted dog poses to SLEAP format!")
            print(f"You can now open {output_file} in SLEAP GUI")
        else:
            print("\nFAILED: Conversion unsuccessful")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()