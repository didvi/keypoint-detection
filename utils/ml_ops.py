import pickle


def load_model(filename):
    params = pickle.load(open(filename, 'rb'))
    return params

def save_model(params, filename):
    pickle.dump(params, open(f"checkpoints/{filename}.p", "wb" ))
