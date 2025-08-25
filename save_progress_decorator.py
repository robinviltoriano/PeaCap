"""
This decorator saves the results of a function call to a file at specified intervals.
"""
import os
import numpy as np
import torch
import pickle
from functools import wraps

def save_progress(save_file_path: str, save_every: int):
    def decorator(func):
        # Load previous results if file exists
        if os.path.exists(save_file_path):
            with open(save_file_path, 'rb') as f:
                results = pickle.load(f)
                
            processed_inputs = set()
            for x in results:
                if isinstance(x['image_id'], (str, int)):
                    processed_inputs.add(x['image_id'])
                elif isinstance(x['image_id'], (list, tuple, np.ndarray)):
                    processed_inputs.update(x['image_id'])
            print(f"Loaded {len(results)} previous results from {save_file_path}")
        else:
            file_name_dir = save_file_path.rsplit('/', 1)[0]
            os.makedirs(file_name_dir, exist_ok=True)
            results = []
            processed_inputs = set()
            print(f"No previous results found, starting fresh.")
        
        @wraps(func)
        def wrapper(*args, **kwargs):

            image_ids = kwargs['image_id']
            
            # if input_key.item() in processed_inputs:
            #     return None  # Skip if already processed
            
            # Normalize ids to list
            if isinstance(image_ids, (str, int)):
                image_ids = [image_ids]
            elif hasattr(image_ids, "tolist"):  # torch.Tensor / np.ndarray
                image_ids = image_ids.tolist()
            elif not isinstance(image_ids, (list, tuple)):
                raise TypeError(f"Unsupported type for image_id: {type(image_ids)}")
            
            # build mask for unprocessed
            unprocessed_mask = [i not in processed_inputs for i in image_ids]
            if not any(unprocessed_mask):
                return None  # all done
            elif all(unprocessed_mask):
                # There are some unprocessed items
                # filter x consistently
                pass
            else: # some done, some not
                x = kwargs["x"]
                new_x = {}
                for k, v in x.items():
                    if hasattr(v, "__getitem__") and len(v) == len(image_ids):
                        if isinstance(v, torch.Tensor):
                            new_x[k] = torch.stack([v[j] for j, keep in enumerate(unprocessed_mask) if keep])
                        elif isinstance(v, list):
                            new_x[k] = [v[j] for j, keep in enumerate(unprocessed_mask) if keep]
                    else:
                        new_x[k] = v
                kwargs["x"] = new_x
                kwargs["image_id"] = torch.tensor([i for i, keep in zip(image_ids, unprocessed_mask) if keep])
            
            # Call the function to process all or some of the inputs
            result = func(*args, **kwargs)
            results.append(result)

            if len(results) % save_every == 0 and len(results) > 0:
                with open(save_file_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Save result to {save_file_path} with number of rows: {len(results)}")

            return result
        wrapper._final_save = lambda: pickle.dump(results, open(save_file_path, 'wb'))
        return wrapper
    return decorator
