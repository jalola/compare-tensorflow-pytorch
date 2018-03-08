import numpy as np
import scipy.misc
import time
from os import listdir, remove
from os.path import isfile, join
import pdb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
DIM=64

def make_generator(mypath, n_files, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        #files = range(n_files)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(onlyfiles)
        epoch_count[0] += 1
        for n, i in enumerate(onlyfiles):
            try:
            	image = scipy.misc.imread("{}/{}".format(mypath, i), mode='RGB')
		#pdb.set_trace()
		assert(len(image.shape)==3)
		assert(image.shape[0]==64 and image.shape[1]==64 and image.shape[2]==3)
            	images[n % batch_size] = image.transpose(2,0,1)
            	if n > 0 and n % batch_size == 0:
                	yield (images,)
	    except Exception as e:
                print("Error reading image")
		print(e)
		print("{}{}".format(mypath,i))
		#remove("{}/{}".format(mypath,i))
		continue
    return get_epoch

def load(batch_size, data_dir='/home/ishaan/data/imagenet64'):
    return (
        make_generator(data_dir+'/', 1281149, batch_size),
        make_generator('/root/cuckccc/Data/valid64', 49999, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
