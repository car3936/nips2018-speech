import os
import subprocess
import scipy.io.wavfile as wav
import sys, getopt


# Put python file into ...\wsj folder and run.
# sph2pipe.exe should be in same folder.

converter_path = 'sph2pipe_v2.5/sph2pipe.exe'
train284_path = 'train_si284.list'
trainall_path = 'train_all.list'
test92_path = 'test_eval92.list'
dev93_path = 'test_dev93.list'

count = 0



def generate_wave(base_path, dst_path):
    # Generate train_si284 wave file
    counter = 0
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))
    f_train_all_list = open(dst_path+'train_all_wav.list', 'w')
    f_eval_list = open(dst_path+'test_dev93_wav.list', 'w')
    f_test_list = open(dst_path+'test_eval92_wav.list', 'w')
    
    f_train_all_label = open(dst_path+'train_all_wav.trans', 'w')
    f_eval_label = open(dst_path+'test_dev93_wav.trans', 'w')
    f_test_label = open(dst_path+'test_eval92_wav.trans', 'w')

    train_all_label = dict()
    eval_label = dict()
    test_label = dict()
    with open(base_path + 'train_all.trans') as f:
        for line in f:
            l = line.split(' ')
            id = l[0]
            trans = " ".join(l[1:])
            train_all_label[id] = trans

    with open(base_path + 'test_dev93.trans') as f:
        for line in f:
            l = line.split(' ')
            id = l[0]
            trans = " ".join(l[1:])
            eval_label[id] = trans
    
    with open(base_path + 'test_eval92.trans') as f:
        for line in f:
            l = line.split(' ')
            id = l[0]
            trans = " ".join(l[1:])
            test_label[id] = trans
   
    with open(base_path + 'train_all.list') as tf:
        for line in tf:
            filename = line.strip('\n')
            id, path = filename.split(' ')
            path = path[2:]
            wav_name = path.split('.')[0]+'.wav'
            if id in train_all_label.keys() and train_all_label[id].find(' ') != -1:
                f_train_all_list.write(wav_name+'\n')
                f_train_all_label.write(train_all_label[id])
            else:
                continue
            dst_file = dst_path +wav_name
            if not os.path.exists(os.path.dirname(dst_file)):
                os.makedirs(os.path.dirname(dst_file))
            command = './sph2pipe -f wav ' + base_path + path + ' ' + dst_file
            print('command : ', command)
            subprocess.call(command, shell = True)
            #if counter == 49:
            #    break
            counter+=1
    
    counter = 0
    with open(base_path + 'test_dev93.list') as tf:
        for line in tf:
            filename = line.strip('\n')
            id, path = filename.split(' ')
            path = path[2:]
            wav_name = path.split('.')[0]+'.wav'
            if id in eval_label.keys() and eval_label[id].find(' ') != -1:
                f_eval_list.write(wav_name+'\n')
                f_eval_label.write(eval_label[id])
            else:
                continue
            dst_file = dst_path +wav_name
            if not os.path.exists(os.path.dirname(dst_file)):
                os.makedirs(os.path.dirname(dst_file))
            command = './sph2pipe -f wav ' + base_path + path + ' ' + dst_file
            print('command : ', command)
            subprocess.call(command, shell = True)
            #if counter == 49:
            #    break
            counter+=1

    counter = 0
    with open(base_path + 'test_eval92.list') as tf:
        for line in tf:
            filename = line.strip('\n')
            id, path = filename.split(' ')
            path = path[2:]
            wav_name = path.split('.')[0]+'.wav'
            if id in test_label.keys() and test_label[id].find(' ') != -1:
                f_test_list.write(wav_name+'\n')
                f_test_label.write(test_label[id])
            else:
                continue
            dst_file = dst_path +wav_name
            if not os.path.exists(os.path.dirname(dst_file)):
                os.makedirs(os.path.dirname(dst_file))
            command = './sph2pipe -f wav ' + base_path + path + ' ' + dst_file
            print('command : ', command)
            subprocess.call(command, shell = True)
            #if counter == 49 :
            #    break
            counter+=1



    f_train_all_list.close()
    f_eval_list.close()
    f_test_list.close()
    f_train_all_label.close()
    f_eval_label.close()
    f_test_label.close()


if __name__ == '__main__':
    base_path =""
    dst_path ="./data_example/"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["inPath=", "outPath="])
    except getopt.GetoptError:
        print('wavgen.py -i <input_path> -o <output_path>')

    for opt, arg in opts:
        if opt == '-h':
            print('wavgen.py -i <input_path> -o <output_path>')
            sys.exit()

        elif opt in ('-i', '--inPath'):
            base_path = arg
        elif opt in ('-o', '--outPath'):
            dst_path = arg

    generate_wave(base_path, dst_path)    
