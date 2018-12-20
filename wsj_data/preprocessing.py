import numpy as np
from scipy.io import wavfile


def framing(signal, frame_length, frame_step, window_func=lambda x: np.ones((x,))):
    """Frame a signal into overlapping frames.
    :param signal: the audio signal to frame.
    :param frame_length: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param window_func: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_length.
    """
    signal_length = len(signal)
    num_frames = 1 + (signal_length - frame_length) // frame_step

    frames = np.zeros((num_frames, frame_length))
    for index in range(num_frames):
        frames[index] = np.asarray(signal[index * frame_step: index * frame_step + frame_length],
                                   dtype='float32') * window_func(frame_length)
    return frames


def get_magnitude(frames, num_fft):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param num_fft: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, num_fft)
    return np.absolute(complex_spec)


def get_power(frames, num_fft):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param num_fft: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    #a = get_magnitude(frames, num_fft)
    #b = np.square(a)
    #print('max : ', np.max(a))
    #print('min : ', np.min(a))
    #print('sq max : ', np.max(b))
    #print('sq min : ', np.min(b))
    #print(a.shape)
    #print(b.shape)
    #return b/num_fft
    return np.square(get_magnitude(frames, num_fft) / np.sqrt(num_fft))


def get_log_power(frames, num_fft, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param num_fft: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = get_power(frames, num_fft)
    ps[ps <= 1e-10] = 1e-10
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def pre_emphasis(signal, coefficient=0.95):
    """perform pre_emphasis on the input signal.

    :param signal: The signal to filter.
    :param coefficient: The pre_emphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    signal = np.asarray(signal, dtype='float32')
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def get_delta(frames, num=2):
    num_frames = len(frames)
    denominator = 2 * sum([i ** 2 for i in range(1, num + 1)])
    delta_frames = np.zeros_like(frames, dtype=frames.dtype)
    padded_frames = np.pad(frames, ((num, num), (0, 0)), mode='edge')
    for index in range(num_frames):
        delta_frames[index] = np.dot(np.arange(-num, num + 1),
                                     padded_frames[index:index + 2 * num + 1]) / denominator
    return delta_frames


def cepstral_mean_normalization(frames, num_before=150, num_after=150):
    num_frames = len(frames)
    new_frames = np.zeros_like(frames, dtype=frames.dtype)
    for index in range(num_frames):
        if index <= num_before:
            mean = np.mean(frames[0:num_before + num_after], axis=0)
        elif num_before < index and index + num_after <= num_frames:
            mean = np.mean(frames[index - num_before: index + num_after], axis=0)
        else:
            mean = np.mean(frames[-(num_before + num_after):], axis=0)
        new_frames[index] = (frames[index] - mean)
    return new_frames


def extract_filter_bank(signal, mel_filter_bank,
                        frame_length=400, frame_step=160,
                        num_fft=512, pre_emphasis_coef=0.95, window_func=np.hamming):

    signal = pre_emphasis(signal, pre_emphasis_coef)
    frames = framing(signal, frame_length, frame_step, window_func)
    power = get_power(frames, num_fft)
    energy = np.sum(power, axis=1)
    energy[energy <= 1e-10] = 1e-10

    filter_bank = np.dot(power, mel_filter_bank.T)
    filter_bank[filter_bank <= 1e-10] = 1e-10

    return filter_bank, energy


def extract_log_filter_bank(signal, mel_filter_bank,
                            frame_length=400, frame_step=160,
                            num_fft=512, pre_emphasis_coef=0.95, window_func=np.hamming):
    filter_bank, energy = extract_filter_bank(signal, mel_filter_bank, frame_length, frame_step,
                                              num_fft, pre_emphasis_coef, window_func)
    return np.log(filter_bank), np.log(energy)


def get_data_x(base_path, file_name, fb):
    """ 
    :return: np.array(n_frame, 40, 3), preprocessed data for one file 
    """
    _, sig = wavfile.read(base_path+file_name)
    feature, _ = extract_log_filter_bank(sig, fb)
    feature_delta = get_delta(feature, 2)
    feature_delta_delta = get_delta(feature_delta, 2)
    data = np.asarray([feature, feature_delta, feature_delta_delta], dtype='float32')
    data = np.transpose(data, [1,2,0])
    return data




def get_mean_var(base_path, list_name, dst_name, fb_file = 'data/fb.npy'):
    """
    this function saves mean and var tensor
    base_path : string, base file path
    list_name : string, list file that contains target file path
    dst_name  : string, save file destination. dst_name_mean.py and dst_name_var.py will be created
    """
    
    fb = np.load(fb_file)
    with open(base_path + list_name, 'r') as f:
        list_lines = f.readlines()


    n_sum = np.zeros((3,40), dtype='float32')
    n_square_sum = np.zeros((3,40), dtype='float32')

    n_file = len(list_lines)
    n_chunk = 5000.0
    n_chunk_square = 20000.0

    print('total file : ',n_file)
    n_frame = 0

    for i in range(n_file):
    #for i in range(15000,16000):
        #print(i,'th')
        l = list_lines[i]
        wav_path = base_path + l[:-1]
        _, sig = wavfile.read(wav_path)
        feature, _ = extract_log_filter_bank(sig, fb)
        feature_delta = get_delta(feature, 2)
        feature_delta_delta = get_delta(feature_delta, 2)
        data = np.asarray([feature, feature_delta, feature_delta_delta], dtype = 'float32')

        n_sum += np.sum(data, axis=1) / n_chunk
        n_square_sum += np.sum(np.multiply(data, data), axis=1) / n_chunk_square
        n_frame+=data.shape[1]
        #print(data.shape)
        #print(np.sum(data, axis=1))

        if(i % 1000 == 0):
            print('---------',i,'th----------- ')
            print('sum')
            print('min : ', np.min(n_sum))
            print('max : ', np.max(n_sum))
            print('square sum')
            print('min : ', np.min(n_square_sum))
            print('max : ', np.max(n_square_sum))

    n_denom = n_frame / n_chunk
    n_denom_square = n_frame / n_chunk_square
    n_mean = n_sum / n_denom
    n_var = (n_square_sum / n_denom_square) - np.multiply(n_mean, n_mean)

    print('----------final mean----------')
    print(n_mean)
    print('----------final var-----------')
    print(n_var)

    np.save(dst_name+'_mean.npy', n_mean)
    np.save(dst_name+'_var.npy', n_var)
    print('final result saved at ',dst_name)
    
def get_frame_list(base_path, list_name, trans_name, dst_name, fb_file = 'data/fb.npy'):
    fb = np.load(fb_file)
    with open(base_path + list_name, 'r') as f:
        list_lines = f.readlines()

    with open(base_path + trans_name, 'r') as f:
        trans_lines = f.readlines()

    n_file = len(list_lines)

    under_400 = []
    under_800 = []
    under_1200 = []


    #print('total file : ', n_file)

    for i in range(n_file):
        l = list_lines[i]
        t = trans_lines[i]
        wav_path = base_path + l[:-1]
        _, sig = wavfile.read(wav_path)

        n_frame = 1 + np.floor((len(sig) - 400) / 160).astype('int32')
        n_frame_compressed = np.ceil(n_frame/4).astype('int32')
        if(i%1000) == 0:
            print(i,'th done')
        if len(t) > n_frame_compressed:
            print(i+1,'th sentence err')

        if n_frame < 400 :
            under_400.append(i)

        if n_frame < 800 :
            under_800.append(i)

        if n_frame < 1200 :
            under_1200.append(i)

    under_400 = np.asarray(under_400, dtype='int32')
    under_800 = np.asarray(under_800, dtype='int32')
    under_1200 = np.asarray(under_1200, dtype='int32')

    np.save(dst_name+'_under_400.npy', under_400)
    np.save(dst_name + '_under_800.npy', under_800)
    np.save(dst_name + '_under_1200.npy', under_1200)

    print('summary')
    print('n_under 400 :', under_400.shape)
    print('n_under 800 :', under_800.shape)
    print('n_under 1200 :', under_1200.shape)

def check_valid_data(base_path, list_name, trans_name):
    with open(base_path + list_name, 'r') as f:
        list_lines = f.readlines()

    with open(base_path + trans_name, 'r') as f:
        trans_lines = f.readlines()

    n_file = len(list_lines)
    err_list = []
    for i in range(n_file):
        l = list_lines[i]
        t = trans_lines[i]
        wav_path = base_path + l[:-1]
        _, sig = wavfile.read(wav_path)

        n_frame = 1+np.floor((len(sig) - 400) / 160).astype('int32')
        n_frame_compressed = np.ceil(n_frame/4).astype('int32')
        if(i % 1000) == 0:
            print(i, 'th done')
        if len(t) > n_frame_compressed:
            print(i+1,'th sentence err')
            print('n_frame : ', n_frame_compressed, ', n_label : ', len(t))
            err_list.append(i+1)

    print(err_list)
if __name__ == '__main__':
    base_path = '/home/yhboo/skt/wavectrl/raw_wav/'
    fb_file = './data/fb.npy'
    list_name = 'train_all.list'
    trans_name = 'train_all.trans'

    # list_name = 'test_eval92.list'
    # trans_name = 'test_eval92.trans'

    dst_name = 'data/train_all_skip'
    #get_mean_var(base_path, list_name, dst_name, fb_file)
    get_frame_list(base_path, list_name, trans_name, dst_name, fb_file)
    #check_valid_data(base_path, list_name, trans_name)


