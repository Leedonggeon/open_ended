from utils import *
from data_loader.modules_language import get_tokenizer
import data_loader.preprocess_script_vip_gpt2 as open_ended
import data_loader.preprocess_script as multiple
#from data_loader.data_loaders import line_to_words, words_to_indices, line_to_indices

sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'

int_dtype = torch.long
float_dtype = torch.float

speaker_name = [
    'None', # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]

speaker_index = {name: index for index, name in enumerate(speaker_name)}

def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


class PredictDataLoader:

    def __init__(self, args, tokenizer = None, vocab = None, device = None):
        # tokenizer for open_ended
        self.args = args
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        # tokenizer for split_tool
        self.split_tool, _ = get_tokenizer(args)

        self.visual_datas = self.load_visual_data()
        
        if tokenizer != None:
            self.pad_index = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
            self.eos_index = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[0])
        else:
            self.pad_index = self.vocab.stoi.get(pad_token)
            self.eos_index = self.vocab.stoi.get(eos_token)
        self.none_index = 0
        self.max_sen_len = args['max_word_per_sentence'] 
        self.empty_sub = '.'
        self.visual_pad = [self.none_index, self.pad_index, self.pad_index]

        self.image_dim = args['image_dim']


        self.image_dt = load_pickle('/data/dataset/AnotherMissOh/AnotherMissOh_images/cache_v3/processed_video_vip.pickle')
        self.image_dt_open = load_pickle('/data/dataset/AnotherMissOh/AnotherMissOh_images/cache_v3/processed_video_vip_open.pickle')
    
    def get_item(self, vid, question, answers = None):
        # que, vgraphs, spkr, script
        data = self.get_data(vid, question, answers)
        if answers == None:
            input_ids, token_type_ids = data_for_gpt(data, self.tokenizer, meta = True, vip = True)
            return input_ids, token_type_ids
        else:
            return data
    
    def get_data(self, vid, question, answer = None):
        data = {}
        subtitle = self.get_subtitle(vid, answer)
        data['spkr'], data['script'] = self.get_script(subtitle)
    
        if answer is None:
            #visual_data = self.visual_datas[vid]
            #data['vgraphs'] = self.get_meta_data(visual_data, self.args['visual_type'])
            _, data['vgraphs'] = self.get_bbft(vid, self.image_dt_open)
            data['que'] = self.get_question(question)
        else:
            data['que'], data['ans'] = self.get_qa(question, answer)
            data ['bbfts'], data['vgraphs'] = self.get_bbft(vid, self.image_dt)
            data = self.data_for_multiple(data)
            
        return data

    def data_for_multiple(self, data):
        if self.args['cc_qa']:
            qa_concat = [[data['que'][j] + data['ans'][j][i] for i in range(5)] for j in range(len(data['que']))]
            qa_concat, _, qa_concat_l = pad3d(qa_concat, self.pad_index, int_dtype)
            data['qa'] = qa_concat.to(self.device)
            data['qa_l'] = qa_concat_l.to(self.device)
        else:
            que, que_l = pad2d(data['que'], self.pad_index, int_dtype)
            ans, _, ans_l = pad3d(data['ans'], self.pad_index, int_dtype)
            data['que'] = que.to(self.device)
            data['que_l'] = que_l.to(self.device)
            data['ans'] = ans.to(self.device)
            data['ans_l'] = ans_l.to(self.device)
        
        data['spkr'] = [data['spkr']]
        data['script'] = [data['script']]
        if self.args['script_type'] == 'word':
            spkr, spkr_l = pad2d(data['spkr'], self.none_index, int_dtype)
            sub, sub_l = pad2d(data['script'], self.pad_index, int_dtype)
            sub_l_l = None
        elif self.args['script_type'] == 'sentence':
            spkr, spkr_l = pad2d(data['spkr'], self.none_index, int_dtype)
            sub, sub_l, sub_l_l = pad3d(data['script'], self.pad_index, int_dtype)

        data['spkr'] = spkr.to(self.device)
        data['sub'] = sub.to(self.device)
        data['sub_l'] = sub_l.to(self.device)
        data['sub_l_l'] = sub_l_l.to(self.device)

        data['bbfts'] = [data['bbfts']]
        data['vgraphs'] = [data['vgraphs']]
        if self.args['visual_type'] == 'frame':
            bbfts, bbfts_l = pad2d(data['bbfts'], 0, float_dtype, reshape3d=True, last_dim=self.image_dim)
            bbfts_l_l = None
            vgraphs, vgraphs_l = pad2d(data['vgraphs'], self.visual_pad, int_dtype)
        elif self.args['visual_type'] == 'shot':
            bbfts, bbfts_l, bbfts_l_l = pad3d(data['bbfts'], 0, float_dtype, reshape4d=True, last_dim=self.image_dim)
            vgraphs, vgraphs_l, _ = pad3d(data['vgraphs'], self.visual_pad, int_dtype, reshape4d=True, last_dim=3)

        data['bbfts'] = bbfts.to(self.device)
        data['bbfts_l'] = bbfts_l.to(self.device)
        data['bbfts_l_l'] = bbfts_l_l.to(self.device)
        data['vmeta'] = vgraphs.to(self.device)
    
        return data
    
    def load_subtitle(self, vid, subtitle_path = '/data/dataset/AnotherMissOh/AnotherMissOh_script.json'):
        with open(subtitle_path, 'r') as f:
            subtitles = json.load(f)
        subtitle = '.'
        if vid in subtitles:
            subtitle = subtitles[vid]
        
        return subtitle
    
    def get_subtitle(self, vid, answer):
        subtitle = self.load_subtitle(vid)

        if subtitle != self.empty_sub:
            subtitle['et'] = float(subtitle['et'])
            subtitle['st'] = float(subtitle['st'])
         
            new_subs = []
            for sub in subtitle['contained_subs']:
                sub['et'] = float(sub['et'])
                sub['st'] = float(sub['st'])
                if answer == None:
                    sub['speaker'] = sub['speaker']
                    split_subs = open_ended.split_subtitle(sub, self.tokenizer, self.split_tool, to_indices=True)
                else:
                    none_index = speaker_index['None'] 
                    sub['speaker'] = none_index if sub['speaker'] == '' else speaker_index[sub['speaker']]
                    word2idx = self.vocab.stoi
                    split_subs = multiple.split_subtitle(sub, self.split_tool, to_indices=True, word2idx=word2idx)
                    print('split_subs : ', split_subs)
                new_subs.extend(split_subs)
            subtitle['contained_subs'] = new_subs
        return subtitle
    
    def get_script(self, subtitle):
        spkr_of_sen_l = []  # list of speaker of subtitle sentences
        sub_in_sen_l = []   # list of subtitle sentences
    
        max_sentence = self.args['max_sentence_per_scene']
        if subtitle != self.empty_sub:  # subtitle exists
            subs = subtitle["contained_subs"]
            n_sentence = 1
    
            for s in subs:
                if n_sentence > max_sentence:
                    break
                n_sentence = n_sentence + 1
                spkr = s["speaker"]
                utter = s["utter"]
                spkr_of_sen_l.append(spkr)
                if len(utter) > self.max_sen_len:
                    del utter[self.max_sen_len:]
                    utter[-1] = self.eos_index
                if self.args['cc_spkr']:
                    utter = [spkr] + utter
                sub_in_sen_l.append(utter)
        else:  # No subtitle
            spkr_of_sen_l.append(self.none_index)  # add None speaker
            sub_in_sen_l.append([self.pad_index])  # add <pad>
    
        return spkr_of_sen_l, sub_in_sen_l
    
    def get_bbft(self, vid, image_dt, flatten=False):
        bb_features = []
        visual_graphs = []
    
        shot = get_shot_id(vid)
        shot_contained = self.image_dt[vid[:19]] if shot == 0 else {vid[:24]: self.image_dt[vid[:19]][vid[:24]]}
    
        max_frame_per_shot = self.args['max_frame_per_shot']
        max_shot_per_scene = self.args['max_shot_per_scene']
    
        shot_num = 1
        for shot_vid, shot in shot_contained.items():
            if shot_num > max_shot_per_scene:
                break
            shot_num = shot_num + 1
            bb_feature = []
            vis_graph = []
    
            frame_num = 1
            # if len(shot.keys()) > max_frame_per_shot:
            #    np.random.uniform(0, len(shot.keys()), max_frame_per_shot)
            # TODO: frame sampling
            for frame_id, frame in shot.items():
                if frame_num > max_frame_per_shot:
                    break
                frame_num = frame_num + 1
                if self.args['remove_metadata']:
                    bb_feature.extend([frame['full_image']])
                else:
                    bb_feature.extend(frame['person_fulls'])
                vis_graph.extend(frame['persons'])
    
            if not bb_feature:
                vis_graph = self.visual_pad
                bb_feature = [np.zeros(self.image_dim)]
            bb_feature = np.reshape(np.concatenate(bb_feature), (-1))
            vis_graph = np.reshape(vis_graph, (-1))
            bb_features.append(bb_feature)
            visual_graphs.append(vis_graph)
    
        return bb_features, visual_graphs
    
    def load_visual_data(self, visual_data_path = '/data/dataset/AnotherMissOh/AnotherMissOh_Visual_v6.json'):
#    def load_visual_data(self, visual_data_path = '/data/dataset/AnotherMissOh/AnotherMissOh_Visual.json'):
        visual_datas = read_json(visual_data_path)
        with open(visual_data_path, 'r') as f:
            visual_datas = json.load(f)
       
        return visual_datas
    
    def get_meta_data(self, visual_datas, visual_type = "shot"):
        meta_datas = []
        for v in visual_datas:
            persons = v['persons']
            for p in persons:
                person_id = p['person_id']
                #### Fix Me ####
                person_id_idx = self.none_index if person_id == '' else tokenize(person_id, self.tokenizer)
        
                behavior = p['person_info']['behavior'].lower()
                behavior_idx = self.pad_index if behavior == '' else tokenize(behavior.split()[0], self.tokenizer) 
        
                emotion = p['person_info']['emotion'].lower()
                emotion_idx = self.pad_index if emotion == '' else tokenize(emotion, self.tokenizer)
                
                meta_data = [person_id_idx, behavior_idx, emotion_idx]
                meta_datas.append(meta_data) 
    
        if visual_type == 'frame':
            vmetas = np.concatenate(vmetas, axis=0)
    
        return meta_datas
    
    def get_question(self, input_que):
        question = [tokenize(input_que, self.tokenizer)]
        return question
    
    
    #### for multiple choice ####
    def get_qa(self, question, answers):
        word2idx = self.vocab.stoi
        que = multiple.line_to_indices(question, self.split_tool, word2idx)
        answers = [multiple.line_to_indices(line, self.split_tool, word2idx) for line in answers]
        return [que], [answers]
    


