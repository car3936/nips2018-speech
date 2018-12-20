import tgt
import numpy as np

def test():
    boundary = get_boundary_from_segmentation('40dc020s.TextGrid')
    print(boundary)



def get_boundary_from_segmentation(tg_file):
    tg = tgt.read_textgrid(tg_file)
    phone_segment = tg.get_tier_by_name('phones')

    boundary = [l._get_start_time() for l in phone_segment._objects] 
    boundary = np.array(boundary)

    return boundary
    
if __name__ == '__main__':
    test()
