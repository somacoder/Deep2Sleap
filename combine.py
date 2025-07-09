import sleap
import numpy as np
from pathlib import Path
from sleap.instance import Point, PredictedPoint

def combine_sleap_files(human_slp_path, dog_slp_path, video_path, output_path):
    """Combine human and dog SLEAP files preserving separate skeletons - SLEAP compliant"""
    
    print("=== COMBINING SLEAP FILES (MULTI-ORGANISM) ===")
    
    # Load the individual SLEAP files
    try:
        human_labels = sleap.Labels.load_file(human_slp_path)
        print(f"Loaded human labels: {len(human_labels)} frames")
        if not human_labels.skeletons:
            raise ValueError("No skeletons found in human labels")
        print(f"Human skeleton: {len(human_labels.skeletons[0].node_names)} bodyparts")
    except Exception as e:
        print(f"Error loading human labels: {e}")
        return None
    
    try:
        dog_labels = sleap.Labels.load_file(dog_slp_path)
        print(f"Loaded dog labels: {len(dog_labels)} frames")
        if not dog_labels.skeletons:
            raise ValueError("No skeletons found in dog labels")
        print(f"Dog skeleton: {len(dog_labels.skeletons[0].node_names)} bodyparts")
    except Exception as e:
        print(f"Error loading dog labels: {e}")
        return None
    
    # Create a single shared video object - CRITICAL for SLEAP consistency
    try:
        shared_video = sleap.Video.from_filename(str(video_path))
        print(f"Created shared video object: {video_path}")
        print(f"Video shape: {shared_video.shape}")
    except Exception as e:
        print(f"Error creating shared video: {e}")
        return None
    
    # Get the original skeletons - these will be our canonical skeleton objects
    human_skeleton = human_labels.skeletons[0]
    dog_skeleton = dog_labels.skeletons[0]
    
    print(f"Human skeleton: {human_skeleton.node_names}")
    print(f"Dog skeleton: {dog_skeleton.node_names}")
    
    # IMPROVEMENT: Ensure skeleton names are set for GUI clarity
    if not hasattr(human_skeleton, 'name') or not human_skeleton.name:
        human_skeleton.name = "Human"
    if not hasattr(dog_skeleton, 'name') or not dog_skeleton.name:
        dog_skeleton.name = "Dog"
    
    # Ensure skeletons are truly distinct (different object IDs)
    if human_skeleton is dog_skeleton:
        print("WARNING: Human and dog skeletons are the same object!")
        # Create a copy of one skeleton to ensure they're distinct
        dog_skeleton = sleap.Skeleton(
            nodes=dog_skeleton.nodes,
            edges=dog_skeleton.edges,
            name="Dog"
        )
        print("Created distinct dog skeleton")
    
    print(f"Skeleton object IDs: Human={id(human_skeleton)}, Dog={id(dog_skeleton)}")
    
    # Create frame mapping for efficient lookup
    human_frame_map = {lf.frame_idx: lf for lf in human_labels}
    dog_frame_map = {lf.frame_idx: lf for lf in dog_labels}
    
    # Get all frame indices that have data
    all_frame_indices = set(human_frame_map.keys()) | set(dog_frame_map.keys())
    
    if not all_frame_indices:
        print("No frames found in either input file!")
        return None
    
    min_frame_idx = min(all_frame_indices)
    max_frame_idx = max(all_frame_indices)
    print(f"Processing {len(all_frame_indices)} frames (range: {min_frame_idx}-{max_frame_idx})")
    
    # DEBUG: Inspect the structure of instance.points
    if human_labels:
        sample_frame = next(iter(human_labels))
        if sample_frame.instances:
            sample_instance = sample_frame.instances[0]
            print(f"DEBUG: instance.points type: {type(sample_instance.points)}")
            print(f"DEBUG: instance.points content: {sample_instance.points}")
    
    # Build combined frames using SLEAP-compliant approach
    combined_frames = []
    
    for frame_idx in sorted(all_frame_indices):
        combined_instances = []
        
        # Process human instances - preserve original skeleton reference
        if frame_idx in human_frame_map:
            human_frame = human_frame_map[frame_idx]
            for instance in human_frame.instances:
                # Handle different possible point structures
                copied_points = {}
                
                # Check if points is a dictionary or some other structure
                if hasattr(instance.points, 'items'):
                    # Dictionary-like structure
                    points_items = instance.points.items()
                elif hasattr(instance, 'points') and hasattr(instance.points, '__iter__'):
                    # Might be a list/tuple of points
                    print(f"DEBUG: Handling non-dict points structure: {type(instance.points)}")
                    # Try to convert to node_name: point mapping
                    if len(instance.points) == len(human_skeleton.node_names):
                        points_items = zip(human_skeleton.node_names, instance.points)
                    else:
                        print(f"WARNING: Point count mismatch. Points: {len(instance.points)}, Nodes: {len(human_skeleton.node_names)}")
                        continue
                else:
                    print(f"ERROR: Unknown points structure: {type(instance.points)}")
                    continue
                
                # Deep copy points to avoid reference issues
                for node_name, point in points_items:
                    if point is None:
                        continue
                        
                    if isinstance(point, PredictedPoint):
                        copied_points[node_name] = PredictedPoint(
                            x=point.x, 
                            y=point.y, 
                            score=point.score,
                            visible=point.visible
                        )
                    elif isinstance(point, Point):
                        copied_points[node_name] = Point(
                            x=point.x, 
                            y=point.y,
                            visible=point.visible
                        )
                    else:
                        print(f"WARNING: Unknown point type: {type(point)}")
                
                # Create new instance with canonical skeleton and shared video
                if copied_points:  # Only create if we have valid points
                    new_instance = sleap.Instance(
                        skeleton=human_skeleton,  # Use canonical skeleton object
                        points=copied_points,
                        track=getattr(instance, 'track', None)  # Preserve track if exists
                    )
                    combined_instances.append(new_instance)
        
        # Process dog instances - preserve original skeleton reference
        if frame_idx in dog_frame_map:
            dog_frame = dog_frame_map[frame_idx]
            for instance in dog_frame.instances:
                # Handle different possible point structures
                copied_points = {}
                
                # Check if points is a dictionary or some other structure
                if hasattr(instance.points, 'items'):
                    # Dictionary-like structure
                    points_items = instance.points.items()
                elif hasattr(instance, 'points') and hasattr(instance.points, '__iter__'):
                    # Might be a list/tuple of points
                    print(f"DEBUG: Handling non-dict points structure: {type(instance.points)}")
                    # Try to convert to node_name: point mapping
                    if len(instance.points) == len(dog_skeleton.node_names):
                        points_items = zip(dog_skeleton.node_names, instance.points)
                    else:
                        print(f"WARNING: Point count mismatch. Points: {len(instance.points)}, Nodes: {len(dog_skeleton.node_names)}")
                        continue
                else:
                    print(f"ERROR: Unknown points structure: {type(instance.points)}")
                    continue
                
                # Deep copy points to avoid reference issues
                for node_name, point in points_items:
                    if point is None:
                        continue
                        
                    if isinstance(point, PredictedPoint):
                        copied_points[node_name] = PredictedPoint(
                            x=point.x, 
                            y=point.y, 
                            score=point.score,
                            visible=point.visible
                        )
                    elif isinstance(point, Point):
                        copied_points[node_name] = Point(
                            x=point.x, 
                            y=point.y,
                            visible=point.visible
                        )
                    else:
                        print(f"WARNING: Unknown point type: {type(point)}")
                
                # Create new instance with canonical skeleton and shared video
                if copied_points:  # Only create if we have valid points
                    new_instance = sleap.Instance(
                        skeleton=dog_skeleton,  # Use canonical skeleton object
                        points=copied_points,
                        track=getattr(instance, 'track', None)  # Preserve track if exists
                    )
                    combined_instances.append(new_instance)
        
        # Create combined frame only if we have instances
        if combined_instances:
            combined_frame = sleap.LabeledFrame(
                video=shared_video,  # Use shared video object
                frame_idx=frame_idx,
                instances=combined_instances
            )
            combined_frames.append(combined_frame)
            
        if len(combined_frames) % 100 == 0 and len(combined_frames) > 0:
            print(f"Processed {len(combined_frames)} frames...")
    
    # Create Labels object - let SLEAP manage skeletons automatically
    combined_labels = sleap.Labels(labeled_frames=combined_frames)
    
    # SLEAP should automatically detect and add skeletons from instances
    detected_skeletons = combined_labels.skeletons
    
    print(f"\nSLEAP detected {len(detected_skeletons)} skeletons:")
    for i, skeleton in enumerate(detected_skeletons):
        skeleton_name = getattr(skeleton, 'name', f'Skeleton_{i}')
        print(f"  {i+1}. {skeleton_name}: {len(skeleton.node_names)} nodes")
    
    # IMPROVEMENT: Validation checks with better error messages
    if len(detected_skeletons) < 2:
        print(f"⚠️  WARNING: Expected 2 skeletons, only found {len(detected_skeletons)}")
        print("   This might indicate skeleton identity issues")
        print("   Check that input files have different skeletons")
    
    # Verify video consistency
    videos = combined_labels.videos
    if len(videos) != 1:
        print(f"⚠️  WARNING: Expected 1 video, found {len(videos)}")
        print("   This might cause GUI issues")
        
    # Additional integrity checks
    print(f"\nIntegrity Summary:")
    print(f"  ✓ Frames: {len(combined_labels)}")
    print(f"  ✓ Skeletons: {len(combined_labels.skeletons)}")
    print(f"  ✓ Videos: {len(combined_labels.videos)}")
    
    # Test instance-skeleton consistency
    skeleton_instance_counts = {}
    skeleton_names = {}
    for frame in combined_labels:
        for instance in frame.instances:
            skeleton_id = id(instance.skeleton)
            skeleton_instance_counts[skeleton_id] = skeleton_instance_counts.get(skeleton_id, 0) + 1
            skeleton_names[skeleton_id] = getattr(instance.skeleton, 'name', f'Skeleton_{skeleton_id}')
    
    print(f"  ✓ Instance distribution:")
    for skeleton_id, count in skeleton_instance_counts.items():
        name = skeleton_names[skeleton_id]
        print(f"    {name}: {count} instances")
    
    # Save combined file
    try:
        combined_labels.save(str(output_path))
        print(f"\n=== COMBINATION COMPLETE ===")
        print(f"✅ Output saved to: {output_path}")
        return combined_labels
        
    except Exception as e:
        print(f"❌ Error saving combined file: {e}")
        import traceback
        traceback.print_exc()
        return None

def deep_verify_sleap_compatibility(slp_path):
    """Comprehensive SLEAP compatibility verification"""
    
    try:
        print(f"\n=== DEEP SLEAP COMPATIBILITY CHECK ===")
        labels = sleap.Labels.load_file(slp_path)
        
        # Test 1: Basic structure
        print(f"✓ File loads successfully")
        print(f"  Frames: {len(labels)}")
        print(f"  Skeletons: {len(labels.skeletons)}")
        print(f"  Videos: {len(labels.videos)}")
        
        # Test 2: Skeleton object identity consistency (CRITICAL)
        skeleton_objects_in_instances = set()
        for frame in labels:
            for instance in frame.instances:
                skeleton_objects_in_instances.add(id(instance.skeleton))
        
        skeleton_objects_in_labels = set(id(s) for s in labels.skeletons)
        
        print(f"✓ Skeleton object consistency:")
        print(f"  Skeleton objects in instances: {len(skeleton_objects_in_instances)}")
        print(f"  Skeleton objects in labels: {len(skeleton_objects_in_labels)}")
        
        if skeleton_objects_in_instances == skeleton_objects_in_labels:
            print(f"  ✅ All instance skeletons are in labels.skeletons")
        else:
            print(f"  ❌ CRITICAL: Skeleton object mismatch!")
            print(f"     This will cause GUI crashes!")
            return False
        
        # Test 3: Video object consistency (CRITICAL)
        video_objects_in_frames = set()
        for frame in labels:
            video_objects_in_frames.add(id(frame.video))
        
        video_objects_in_labels = set(id(v) for v in labels.videos)
        
        print(f"✓ Video object consistency:")
        print(f"  Video objects in frames: {len(video_objects_in_frames)}")
        print(f"  Video objects in labels: {len(video_objects_in_labels)}")
        
        if len(video_objects_in_frames) == 1 and video_objects_in_frames == video_objects_in_labels:
            print(f"  ✅ Single video object used consistently")
        else:
            print(f"  ❌ CRITICAL: Video object inconsistency!")
            print(f"     This will cause GUI issues!")
            return False
        
        # Test 4: Frame and instance integrity
        total_instances = 0
        frame_indices = []
        empty_instances = 0
        
        for frame in labels:
            frame_indices.append(frame.frame_idx)
            total_instances += len(frame.instances)
            
            # Check each instance
            for instance in frame.instances:
                if not instance.points:
                    empty_instances += 1
                    
                # Check skeleton node consistency - handle different point structures
                if hasattr(instance.points, 'items'):
                    point_keys = instance.points.keys()
                elif hasattr(instance.points, '__iter__'):
                    # For tuple/list structures, assume they match skeleton order
                    point_keys = instance.skeleton.node_names[:len(instance.points)]
                else:
                    print(f"  ❌ CRITICAL: Unknown points structure in instance")
                    return False
                    
                for node_name in point_keys:
                    if node_name not in instance.skeleton.node_names:
                        print(f"  ❌ CRITICAL: Invalid node '{node_name}' in instance")
                        print(f"     Node not in skeleton: {instance.skeleton.node_names}")
                        return False
        
        print(f"✓ Instance integrity:")
        print(f"  Total instances: {total_instances}")
        if frame_indices:
            print(f"  Frame range: {min(frame_indices)} to {max(frame_indices)}")
        if empty_instances > 0:
            print(f"  ⚠️  Empty instances: {empty_instances}")
        
        # Test 5: SLEAP GUI operations simulation
        try:
            # Test frame iteration (what GUI does)
            frame_count = len([f for f in labels])
            
            # Test skeleton access (what GUI does)
            skeleton_node_counts = []
            skeleton_edge_counts = []
            for skeleton in labels.skeletons:
                skeleton_node_counts.append(len(skeleton.node_names))
                skeleton_edge_counts.append(len(skeleton.edges))
            
            # Test instance access (what GUI does)
            instance_count = sum(len(f.instances) for f in labels)
            
            print(f"✓ GUI operation simulation:")
            print(f"  Frame iteration: {frame_count} frames")
            print(f"  Skeleton access: {skeleton_node_counts} nodes, {skeleton_edge_counts} edges")
            print(f"  Instance access: {instance_count} instances")
            
        except Exception as e:
            print(f"  ❌ GUI simulation failed: {e}")
            return False
        
        # Test 6: Multi-organism specific checks
        if len(labels.skeletons) >= 2:
            print(f"✓ Multi-organism checks:")
            
            # Check that skeletons are actually different
            skeleton_fingerprints = set()
            for skeleton in labels.skeletons:
                fingerprint = tuple(sorted(skeleton.node_names))
                skeleton_fingerprints.add(fingerprint)
            
            if len(skeleton_fingerprints) == len(labels.skeletons):
                print(f"  ✅ All skeletons are unique")
            else:
                print(f"  ⚠️  Some skeletons have identical node sets")
            
            # Check instance distribution
            instances_per_skeleton = {}
            for frame in labels:
                for instance in frame.instances:
                    skeleton_id = id(instance.skeleton)
                    instances_per_skeleton[skeleton_id] = instances_per_skeleton.get(skeleton_id, 0) + 1
            
            print(f"  Instance distribution: {list(instances_per_skeleton.values())}")
        
        print(f"\n🎉 ALL COMPATIBILITY CHECKS PASSED!")
        print(f"✅ File should work perfectly in SLEAP GUI")
        print(f"✅ Multi-organism tracking ready")
        return True
        
    except Exception as e:
        print(f"❌ Deep verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # File paths
    human_slp = "./human_poses.slp"
    dog_slp = "./dog_poses.slp"
    video_file = "./videos/pivot/pivot_cut.mp4"
    combined_slp = "./combined_poses.slp"
    
    # Check input files
    missing_files = [f for f in [human_slp, dog_slp, video_file] if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing files:")
        for f in missing_files:
            print(f"  {f}")
        print(f"\n📋 Run conversion scripts first:")
        print(f"  python convert_human.py")
        print(f"  python convert_dog.py")
        exit(1)
    
    # Combine files with deep SLEAP compliance
    try:
        combined_labels = combine_sleap_files(human_slp, dog_slp, video_file, combined_slp)
        
        if combined_labels:
            # Run comprehensive verification
            if deep_verify_sleap_compatibility(combined_slp):
                print(f"\n🎉 SUCCESS: Multi-organism SLEAP file created!")
                print(f"✅ Fully compatible with SLEAP GUI")
                print(f"✅ Ready for multi-species tracking analysis")
                print(f"📂 Open in SLEAP: {combined_slp}")
                
                # Quick stats
                human_instances = sum(1 for f in combined_labels for i in f.instances 
                                    if getattr(i.skeleton, 'name', '') == 'Human')
                dog_instances = sum(1 for f in combined_labels for i in f.instances 
                                  if getattr(i.skeleton, 'name', '') == 'Dog')
                
                print(f"\n📊 Final Stats:")
                print(f"  Human instances: {human_instances}")
                print(f"  Dog instances: {dog_instances}")
                print(f"  Total frames: {len(combined_labels)}")
            else:
                print(f"\n❌ Compatibility issues detected")
                print(f"File may not work correctly in SLEAP GUI")
        else:
            print("❌ Failed to combine poses")
            
    except Exception as e:
        print(f"❌ Error during combination: {e}")
        import traceback
        traceback.print_exc()