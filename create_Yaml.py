import yaml


def create_Yaml(args, name):
    
    if len(args) < 18 or len(args) > 20 or len(args) == 19:
        print(f"Error: not enough/too many arguments. Expected 18 or 20 received {len(args)}")
        return
    else:
        builder  = dict({'model_Name':args[0], 
           'image_Path':args[1],
           'model_Path':args[2],
           'config_Path':args[3],
           'test_Path':args[4],
           'train_Path':args[5],
           'label_Path':args[6],
           'api_Path':args[7],
           'checkpoint_Path':args[8],
           'results_Path':args[9],
           'annotations_Path':args[10],
           'test_Record':args[11],
           'train_Record':args[12],
           'controlled':args[13],
           'base_Resolution':args[14],
           'final_Resolution':args[15],
           'model_Type':args[16],
           'setup':args[17]
           })
        if len(args) == 20:
            builder['test_annotations'] = args[18]
            builder['train_annotations'] = args[19]
        
    with open(f'YAMLs/{str(name)}.yml', 'w') as yaml_file:
        yaml.dump(builder, yaml_file, default_flow_style=False, sort_keys=False)