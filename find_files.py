import os

def get_paths(model_folder):
    model_path = model_folder
    pipeline_path = os.path.join(model_folder, 'pipeline.config').replace("\\","/")
    checkpoint_path = os.path.join(model_folder, 'checkpoint0').replace("\\","/")
    return pipeline_path, checkpoint_path
    
