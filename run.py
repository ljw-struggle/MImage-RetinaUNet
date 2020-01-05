# -*- coding: utf-8 -*-
import os, sys
import configparser

def train():
    config = configparser.RawConfigParser()
    config.readfp(open(r'./configuration.txt'))
    name_experiment = config.get('experiment name', 'name')
    nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

    run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

    # create a folder for the results
    result_dir = name_experiment
    if os.path.exists(result_dir):
        print("Dir already existing")
    elif sys.platform=='win32':
        os.system('mkdir ' + result_dir)
    else:
        os.system('mkdir -p ' +result_dir)
    if sys.platform=='win32':
        os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
    else:
        os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

    if nohup:
        os.system(run_GPU +' nohup python -u ./retinaNN_training.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
    else:
        os.system(run_GPU +' python ./retinaNN_training.py')

def test():
    config = configparser.RawConfigParser()
    config.readfp(open(r'./configuration.txt'))
    name_experiment = config.get('experiment name', 'name')
    nohup = config.getboolean('testing settings', 'nohup')  # std output on log file?

    run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

    # create a folder for the results if not existing already
    result_dir = name_experiment
    if os.path.exists(result_dir):
        pass
    elif sys.platform == 'win32':
        os.system('md ' + result_dir)
    else:
        os.system('mkdir -p ' + result_dir)

    # finally run the prediction
    if nohup:
        os.system(run_GPU + ' nohup python -u ./src/retinaNN_predict.py > ' + './' + name_experiment + '/' + name_experiment + '_prediction.nohup')
    else:
        os.system(run_GPU + ' python ./src/retinaNN_predict.py')