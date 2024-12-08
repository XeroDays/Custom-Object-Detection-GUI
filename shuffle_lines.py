import random
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class shuffle_lines():
    def __init__(self, filepath):
        print(f"FILEPATH {filepath}")
        ext = os.path.splitext(filepath)[-1].lower()
        self.filepath = filepath
        # Now we can simply use == to check for equality, no need for wildcards.
        print(f"Annot extension {ext}")
        if ext == ".csv":
            self.shuffle_csv()
        elif ext == ".txt":
            self.shuffle_txt()
        else:
            logger.error("Annotations File is not type .txt or .cvs")

    def shuffle_txt(self):
        lines = open(self.filepath,'r').readlines()
        random.shuffle(lines)
        open(self.filepath, 'w').writelines(lines)
    
    def shuffle_csv(self):
        df = pd.read_csv(self.filepath, skiprows=0, skip_blank_lines=True)
        shuffled  = df.sample(frac=1)   
        shuffled.to_csv(self.filepath,index=False)


