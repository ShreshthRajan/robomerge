import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from robomerge.ingestion import DROIDIngestion
from robomerge.transform import DataStandardizer
from robomerge.fast_prep import FASTPreprocessor

def inspect_tfrecord(filepath):
    """Debug function to inspect TFRecord content."""
    print(f"Inspecting TFRecord: {filepath}")
    dataset = tf.data.TFRecordDataset(filepath)
    
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\nAvailable features:")
        features = example.features.feature
        for key in features.keys():
            print(f"- {key}")
            
        return example  # Return the first example for inspection

def parse_tfrecord(serialized_example):
    """Parse TFRecord according to DROID schema."""
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())
    features = example.features.feature
    
    try:
        raw_actions = np.array(features['steps/action'].float_list.value, dtype=np.float32)
        actions = raw_actions.reshape(-1, 7)
        
        raw_states = np.array(features['steps/observation/joint_position'].float_list.value, dtype=np.float32)
        states = raw_states.reshape(-1, 7)
        
        if len(actions) != len(states):
            raise ValueError(f"Mismatch in sequence lengths: actions={len(actions)}, states={len(states)}")
        
        print(f"\nValidation - Sequence length: {len(actions)} timesteps")
        print(f"Action dims per step: {actions.shape[1]}")
        print(f"State dims per step: {states.shape[1]}")
        
        images = {}
        if 'steps/observation/wrist_image_left' in features:
            wrist_img = tf.io.decode_jpeg(
                features['steps/observation/wrist_image_left'].bytes_list.value[0]
            ).numpy()
            images['wrist'] = wrist_img
        
        if 'steps/observation/exterior_image_1_left' in features:
            ext_img = tf.io.decode_jpeg(
                features['steps/observation/exterior_image_1_left'].bytes_list.value[0]
            ).numpy()
            images['external'] = ext_img
        
        instruction = ''
        if features['steps/language_instruction'].bytes_list.value:
            instruction = features['steps/language_instruction'].bytes_list.value[0].decode('utf-8')
        
        print("\nExtracted Data:")
        print(f"Actions shape: {actions.shape}")
        print(f"States shape: {states.shape}")
        for k, v in images.items():
            print(f"Image {k} shape: {v.shape}")
        print(f"Instruction length: {len(instruction)}")
            
        return actions, states, images, instruction
        
    except Exception as e:
        print(f"\nDetailed extraction error:")
        print(f"Available keys: {list(features.keys())}")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

def run_pipeline(tfrecord_path: str, output_dir: str):
    """End-to-end pipeline for processing robot data."""
    print(f"\nProcessing TFRecord: {tfrecord_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    h5_output = output_path / "test_episode.h5"

    try:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        for raw_record in raw_dataset.take(1):
            try:
                actions, states, images, instruction = parse_tfrecord(raw_record)
                if actions.size == 0 or states.size == 0:
                    raise ValueError("Empty actions or states array")
                print("Successfully parsed TFRecord")
            except Exception as e:
                print(f"Error parsing TFRecord: {e}")
                import traceback
                traceback.print_exc()
                return
            break

        with h5py.File(h5_output, "w") as f:
            print("Writing to HDF5...")
            f.create_dataset("actions", data=actions)
            f.create_dataset("states", data=states)
            f.create_dataset("timestamps", data=np.arange(len(actions)) / 15.0)
            images_group = f.create_group("images")
            for key, img in images.items():
                images_group.create_dataset(key, data=img)
            f.attrs['language_instruction'] = instruction
            print("Successfully wrote to HDF5")

        ingestion = DROIDIngestion()
        episode = ingestion.load_episode(str(h5_output))
        print("Data ingestion successful.")

        standardizer = DataStandardizer()
        standardized_data = standardizer.standardize_episode(episode)
        print("Data standardization successful.")

        if 'metadata' not in standardized_data:
            standardized_data['metadata'] = {}
        
        preprocessor = FASTPreprocessor()
        fast_data = preprocessor.prepare_episode(standardized_data)
        print("\nProcessed Data Summary:")
        print(f"Number of action chunks: {fast_data.action_chunks.shape[0]}")
        print(f"Chunk size: {fast_data.action_chunks.shape[1]}")
        print(f"Action dimensions: {fast_data.action_chunks.shape[2]}")
        print(f"Available observations: {list(fast_data.observations.keys())}")
        
        print("\nPipeline completed successfully.")

    except Exception as e:
        print(f"Error during pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    data_dir = "/Users/shreshth.rajan/projects/robomerge/data/droid/1.0.0"
    output_dir = "/Users/shreshth.rajan/projects/robomerge/output"

    tfrecord_path = next(Path(data_dir).glob("*.tfrecord-*"), None)
    if tfrecord_path is None:
        print("Error: No TFRecord files found in the directory!")
        exit(1)

    example = inspect_tfrecord(str(tfrecord_path))
    print("\nProceeding with pipeline...")
    run_pipeline(str(tfrecord_path), output_dir)
