from nltk.corpus import brown
import operator

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])

def get_sentences():
    return brown.sents()

def get_sentence_with_word2idx():
    sentences = get_sentences()
    indexed_sentences = []
    
    word2idx = {'START':0,'END':1}
    current_index = 2
    
    for sentence in sentences:
        indexed_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = current_index
                current_index += 1
            indexed_sentence.append(word2idx[word])
        indexed_sentences.append(indexed_sentence)
    
    print ("Vocabulary size:",current_index)
    return indexed_sentences, word2idx


def get_sentence_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
    sentences = get_sentences()
    indexed_sentences = []
    
    word2idx = {'START':0,'END':1}
    idx2word = ['START','END']
    current_index = 2
    word_idx_count = {0:float('inf'), 1:float('inf')}
    
    # create dictionary and the indexed sentences
    for sentence in sentences:
        indexed_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = current_index
                idx2word.append(word)
                current_index += 1
        
            idx = word2idx[word]
            word_idx_count[idx] = word_idx_count.get(idx,0) + 1  # dict.get(key, default = None) .  default − This is the Value to be returned in case key does not exist
            
            indexed_sentence.append(word2idx[word])
        indexed_sentences.append(indexed_sentence)
    
    # settting the frequency of all the words we have to infinity
    # so that they are included when I pick the most
    # common words
    for word in keep_words:
        idx = word2idx[word] 
        word_idx_count[idx] = float('inf')
        
    
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
        old_idx = word2idx[word]
        word2idx_small[word] = idx_new_idx_map[old_idx]
        
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx
    
    # sanity check just  to ensure that things went well
    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    for word in keep_words:
      assert(word in word2idx_small)
    
    indexed_sentences_small = []
    for indexed_sentence in indexed_sentences:
        if len(indexed_sentence)>1:
#            indexed_sentence_small = []
#            for old_idx in indexed_sentence:
#                new_idx = idx_new_idx_map.get(old_idx, unknown) #returns index of 'UKNOWN' if the word is not in top 2000 (or n_vocab) words
#                indexed_sentence_small.append(new_idx)
            indexed_sentence_small = [idx_new_idx_map[old_idx] if old_idx in idx_new_idx_map else unknown for old_idx in indexed_sentence ]            
            indexed_sentences_small.append(indexed_sentence_small)
        
    return indexed_sentences_small, word2idx_small





