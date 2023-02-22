import os
import sys
from pdb import set_trace as bp
import json
import re
import pandas as pd
from copy import deepcopy

def clamp(v, _min, _max):
    return max(_min, min(v, _max))


def generate_csv(duration_file, inp_json, phase, output_csv):
    df1 = pd.read_csv(duration_file,sep=' ',header=None)
    f=open(inp_json,'r')
    d1=json.loads(f.read())

    column_names=['video_id','caption', 'summary','dialog','start','end','duration', 'seq_start', 'seq_end', 'tan_mask', 'phase','idx']
    # df2=pd.DataFrame(columns = column_names, dtype=object)
    ld={item1['image_id'] : item1 for item1 in d1['dialogs']}
    d_list=[]
    c=0
    for index, item in df1.iterrows():
        key = re.sub(r'\.mp4$', '', item[0])
        if key in ld:
            item1 = ld[key]
            d={}
            d['video_id']=key
            d['duration']=item[1]
            d['end'] = item[1]
            d['start'] = 0
            for cap in ['caption', 'summary']:
                d[cap] = 'C: ' + (item1[cap] if cap in item1 else '') + ' CLS'
            d['phase'] = phase

            dialogs = item1['dialog']
            if phase in ('train', 'val'):
                start_idx_e = 1
            else:
                start_idx_e = len(dialogs)

            for idx_e in range(start_idx_e, len(dialogs)+1):
                tmp = deepcopy(d)

                # get dialog (only previous and present)
                idx_s = max(idx_e - 2, 0)
                dialog = dialogs[idx_s:idx_e]
                # generate dialog string
                dialog_str = 'Q: '
                for turn_idx in range(len(dialog)):
                    dialog_str += dialog[turn_idx]['question'] + ' '
                    if turn_idx == len(dialog) - 1:
                        dialog_str += ' A: '
                    if type(dialog[turn_idx]['answer']) == list:
                        dialog_str += dialog[turn_idx]['answer'][0]
                    else:
                        dialog_str += dialog[turn_idx]['answer']
                    dialog_str += ' '
                tmp['dialog'] = dialog_str
                # bp()
                turn = dialog[-1]
                if phase == 'test' or ('reason' not in turn) or (len(turn['reason'])==0):
                    tmp['tan_mask'] = [0]
                    tmp['seq_start'] = [[-1]]
                    tmp['seq_end'] = [[-1]]
                else:
                    mask = 0
                    ss = []
                    es = []
                    for r in turn['reason']:
                        s, e = r['timestamp']
                        s = clamp(s, 0, d['duration'])
                        e = clamp(e, 0, d['duration'])
                        if s > e:
                            s, e = e, s
                        # sometimes start and end are the same
                        if s != e:
                            mask = 1
                        ss.append(s)
                        es.append(e)
                    tmp['tan_mask'] = [mask]
                    tmp['seq_start'] = [ss]
                    tmp['seq_end'] = [es]
                tmp['idx'] = c
                d_list.append(tmp)

                c+=1 

    df2 = pd.DataFrame(d_list, dtype=object)
    df2 = df2.reindex(columns=column_names)
    df2.to_csv(output_csv, sep='\t', index=None)
    # bp()
    if phase in ('train', 'val'):
        if phase == 'train':
            debug = df2.iloc[:100]
        else:
            debug = df2.iloc[:20]
        debug.to_csv(output_csv[:-4]+'_debug.csv', sep='\t', index=None)

if __name__=="__main__":
    generate_csv(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])