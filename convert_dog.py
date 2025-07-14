import h5py
import numpy as np
import pandas as pd
import sleap
from sleap.instance import Point, PredictedPoint
from pathlib import Path

# Configuration: Number of top animals to keep (ranked by detection quality)
TOP_N_ANIMALS = 1  # Change this to keep more animals (e.g., 2, 3, etc.)

h5_file = "./videos/pivot_cut_superanimal_quadruped_fasterrcnn_resnet50_fpn_v2_hrnet_w32.h5"
video_file = "./videos/pivot_cut.mp4"
output_file = "./pivot_cut.slp"

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
    print(f"Keeping top {TOP_N_ANIMALS} animals only")
    
    # Filter out antler bodyparts and non-essential body parts since they're not relevant for dogs
    relevant_bodyparts = [bp for bp in bodyparts if 'antler' not in bp.lower() and 
                         bp not in ['body_middle_left', 'body_middle_right', 'belly_bottom', 
                                   'neck_base', 'throat_base', 'throat_end', 
                                   'mouth_end_left', 'mouth_end_right', 'lower_jaw', 'upper_jaw', 'neck_end',
                                   'left_earbase', 'left_earend', 'right_earbase', 'right_earend',
                                   'tail_base', 'tail_end']]
    
    print(f"Original bodyparts ({len(bodyparts)}): {bodyparts}")
    print(f"Relevant bodyparts ({len(relevant_bodyparts)}): {relevant_bodyparts}")
    
    # Rank animals by detection quality across all frames
    animal_scores = {}
    for animal in animals:
        total_valid_points = 0
        total_confidence = 0
        point_count = 0
        
        for frame_idx in range(len(df)):
            for bodypart in relevant_bodyparts:
                try:
                    likelihood = df.loc[frame_idx, (scorer, animal, bodypart, 'likelihood')]
                    x = df.loc[frame_idx, (scorer, animal, bodypart, 'x')]
                    y = df.loc[frame_idx, (scorer, animal, bodypart, 'y')]
                    
                    if (pd.notna(x) and pd.notna(y) and pd.notna(likelihood) and
                        x > 0 and y > 0 and likelihood > 0):
                        total_valid_points += 1
                        total_confidence += likelihood
                        point_count += 1
                except:
                    continue
        
        # Score = average confidence * number of valid detections
        avg_confidence = total_confidence / max(point_count, 1)
        animal_scores[animal] = avg_confidence * total_valid_points
    
    # Select top N animals
    if TOP_N_ANIMALS == 0:
        # Keep all animals if TOP_N_ANIMALS is 0
        selected_animals = animals
        print(f"\nTOP_N_ANIMALS set to 0 - keeping ALL animals ({len(animals)})")
    else:
        # Select top N animals based on ranking
        top_animals = sorted(animal_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N_ANIMALS]
        selected_animals = [animal for animal, score in top_animals]
        print(f"\nAnimal ranking by detection quality:")
        for animal, score in sorted(animal_scores.items(), key=lambda x: x[1], reverse=True):
            status = "✓ SELECTED" if animal in selected_animals else "✗ skipped"
            print(f"  {animal}: {score:.1f} {status}")
    
    # Create one Track per selected animal so we can carry identity
    tracks = {animal: sleap.Track(name=animal) for animal in selected_animals}

    # Create SLEAP skeleton with only relevant quadruped bodyparts
    skeleton = sleap.Skeleton.from_names_and_edge_inds(
        node_names=relevant_bodyparts,
        edge_inds=[
            # Eyes to nose (head core)
            (relevant_bodyparts.index('left_eye'), relevant_bodyparts.index('nose')),
            (relevant_bodyparts.index('right_eye'), relevant_bodyparts.index('nose')),
            
            # Head to body connection - nose directly to back_base
            (relevant_bodyparts.index('nose'), relevant_bodyparts.index('back_base')),
            
            # Main spine (the core body structure)
            (relevant_bodyparts.index('back_base'), relevant_bodyparts.index('back_middle')),
            (relevant_bodyparts.index('back_middle'), relevant_bodyparts.index('back_end')),
            
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
            
            ## Tail
            #(relevant_bodyparts.index('back_end'), relevant_bodyparts.index('tail_base')),
            #(relevant_bodyparts.index('tail_base'), relevant_bodyparts.index('tail_end')),
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
    confidence_threshold = 0  # Minimum confidence for valid keypoints
    
    for frame_idx in range(len(df)):
        instances = []
        
        # Process only selected animals in this frame
        for animal in selected_animals:  # Changed from all animals
            points = {}
            valid_points = 0
            
            # Process each bodypart
            for bodypart in relevant_bodyparts:
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
            
            if valid_points >= 5:  # Require at least 5 valid keypoints
                # build instance (don't pass unsupported 'score' kwarg)
                instance = sleap.Instance(
                    skeleton=skeleton,
                    points=points
                )

                # assign identity
                instance.track = tracks[animal]
                
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
    # Create labels and register the tracks so identity & scores persist
    for tr in tracks.values():
        labels.tracks.append(tr)
    
    # Save SLEAP project
    try:
        labels.save(str(output_slp_path))
        print(f"\n=== DOG CONVERSION COMPLETE ===")
        print(f"Valid frames: {valid_frames}/{len(df)}")
        print(f"Total instances: {total_instances}")
        print(f"Selected animals: {selected_animals}")
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
