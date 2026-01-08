import h5py
import json

try:
    f = h5py.File("keras_model.h5", mode='r')
    print("Keys in H5:", list(f.keys()))
    
    if 'model_config' in f.attrs:
        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        config = json.loads(config_str)
        print("Model Config Type:", type(config))
        print("Model Class Name:", config.get('class_name'))
        
        # Save config to file for reading
        with open("model_config.json", "w") as jf:
            json.dump(config, jf, indent=2)
        print("Saved model_config.json")
    else:
        print("No model_config found in attributes.")
        
    f.close()
except Exception as e:
    print("Error:", e)
