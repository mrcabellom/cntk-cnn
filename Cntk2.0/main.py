from data_loader.load_data import try_download
from data_loader.save_data import savetxt
from settings import TRAIN_DATA, TEST_DATA
import os


print('Downloading train data')
train = try_download(TRAIN_DATA['URL_IMAGE'],TRAIN_DATA['URL_LABELS'], TRAIN_DATA['SAMPLES'])

print('Downloading test data')
test = try_download(TEST_DATA['URL_IMAGE'],TEST_DATA['URL_LABELS'], TEST_DATA['SAMPLES'])

data_dir = os.path.join('data', "MNIST")

print('Writing train text file...')
savetxt(os.path.join(data_dir, 'Train-28x28_cntk_text.txt'), train)

print('Writing test text file...')
savetxt(os.path.join(data_dir, 'Test-28x28_cntk_text.txt'), test)

print('Done')