from path import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

def get_context(X, w=12):
    N, D = X.shape
    # zero padding
    X = np.r_[np.zeros((w, D)) + X[0], X, np.zeros((w, D)) + X[-1]]
    X = np.array([X[i:i + 2 * w + 1].flatten() for i in range(N)])
    return X

class Data:
    def __init__(self, home_dir):
        self.home_dir = home_dir
        self.fbank_dir = Path(f'{home_dir}timit_fbank')
        self.lab_dir = Path(f"{home_dir}timit_phn/")
        self.phon_dim = 35
        self.scaler = None
        self.cls2sym = np.array(
            ['o', 'p', 'ɹ', 'w', 'ɡ', 'ʊ', 'd', 't', 'a', 'f', 'b', 'i', '.', 'v', 'ŋ', 's', 'n', 'ɑ', 'k', 'ʃ',
             'l', 'e', 'ð', 'ɛ', 'ɜ˞', 'ʌ', 'u', 'æ', 'm', 'z', 'ɾ', 'j', 'θ', 'h', 'ɪ'])

        self.sym2cls = {s: i for i, s in enumerate(self.cls2sym)}

    def cal_scaler(self):
        print('Cal scaler ...')
        train_paths = self.fbank_dir.glob('TRAIN*')
        feat_data = []
        for i, p in enumerate(train_paths):
            feat_data.append(np.load(p))
        feat_data = np.concatenate(feat_data)
        self.scaler = StandardScaler().fit(feat_data)

    def read_data(self, paths, M=12, data_dim=32):
        X_dat = []
        y_dat = []
        for p in paths:
            X = np.load(p)
            X = self.scaler.transform(X)  # standard scaling
            X = get_context(X)  # stacking frames
            X = X.reshape(X.shape[0], M * 2 + 1, data_dim, 1)
            X_dat.append(X)

            lab_path = f"{self.lab_dir}{p.name.replace('.npy','')}"
            lab = np.genfromtxt(lab_path, dtype=str)
            lab = [self.sym2cls[sym] for sym in lab]
            lab = to_categorical(lab, self.phon_dim)  # to one-hot vectors
            y_dat.append(lab)

        X_dat = np.concatenate(X_dat)
        y_dat = np.concatenate(y_dat)
        return X_dat, y_dat

    def create_data(self, kind):
        # cal scaler
        if self.scaler==None:
            self.cal_scaler()
        if kind == 'train':
            print('create train data')
            train_paths = self.fbank_dir.glob('TRAIN*')
            x_train, y_train = self.read_data(train_paths)
            valid_paths = self.fbank_dir.glob('DEV*')
            x_valid, y_valid = self.read_data(valid_paths)
            return (x_train, y_train), (x_valid, y_valid)
        else:
            print('create test data')
            test_paths = self.fbank_dir.glob('TEST*')
            x_test, y_test = self.read_data(test_paths)
            return x_test, y_test


if __name__ == "__main__":
    data = Data('/Users/dintu/work_sp/data/')
    (x_train, y_train), (x_valid, y_valid) = data.create_data('train')
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    x_test, y_test = data.create_data('test')
    print(x_test.shape, y_test.shape)