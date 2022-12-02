import os
import sys
import pdb
import json
import re
import pandas as pd

def clamp(v, _min, _max):
    return max(_min, min(v, _max))


def generate_csv(duration_file, inp_json, phase, output_csv):
    df1 = pd.read_csv(duration_file,sep=' ',header=None)
    f=open(inp_json,'r')
    d1=json.loads(f.read())

    column_names=['video_id','caption','start','end','duration', 'seq_start', 'seq_end', 'train_mask', 'phase','idx']
    df2=pd.DataFrame(columns = column_names, dtype=object)
    ld={item1['image_id'] : item1 for item1 in d1['dialogs']}
    d_list=[]
    c=0
    for index, item in df1.iterrows():
        key = re.sub(r'\.mp4$', '', item[0])
        if key in ld:
            item1 = ld[key]
            d={}
            d['video_id']=key
            d['idx']=c
            d['duration']=item[1]
            d['end'] = item[1]
            d['start'] = 0
            d['caption'] = ' '.join(['Q: ' + item_ins['question'] 
                                  + ' A: ' + (item_ins['answer'][0] if type(item_ins['answer'])==list
                                              else item_ins['answer'])
                                    for item_ins in item1['dialog']])
            if phase in ('train', 'val'):
                masks, starts, ends = [], [], []
                # d['seq_start'], d['seq_end'] = [], []
                for turn in item1['dialog']:
                    if ('reason' not in turn) or \
                       (len(turn['reason']) == 0):
                        masks.append(0)
                        starts.append([-1]) 
                        ends.append([-1])
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
                        masks.append(mask)
                        starts.append(ss)
                        ends.append(es)

                d['train_mask'] = masks
                d['seq_start'] = starts
                d['seq_end'] = ends
            else:
                d['train_mask'] = -1
                d['seq_start'] = -1
                d['seq_end'] = -1

            d['phase'] = phase
            d_list.append(d)
            c+=1 

    df2 = pd.DataFrame(d_list)
    df2 = df2.reindex(columns=column_names)
    df2.to_csv(output_csv, sep='\t', index=None)
    #pdb.set_trace()

if __name__=="__main__":
    generate_csv(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
