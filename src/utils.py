import pickle


def print_devider(title):
  print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))


def load_pickle(fp):
  with open(fp, 'rb') as f:
    return pickle.load(f)
