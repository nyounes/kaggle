import re
import string

import mappings


def remove_spaces(text):
    for space in mappings.spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text


def clean_special_punctuations(text):
    for punc in mappings.special_punc_mappings:
        if punc in text:
            text = text.replace(punc, mappings.special_punc_mappings[punc])
    return text


def clean_number(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'(\d+)(e)(\d+)','\g<1> \g<3>', text)
    return text


def clean_rare_words(text):
    for rare_word in mappings.rare_words_mapping:
        if rare_word in text:
             text = text.replace(rare_word, mappings.rare_words_mapping[rare_word])
    return text


def clean_misspell(text):
    for bad_word in mappings.mispell_dict:
        if bad_word in text:
            text = text.replace(bad_word, mappings.mispell_dict[bad_word])
    return text


def spacing_punctuation(text):
    regular_punct = list(string.punctuation)
    all_punct = list(set(regular_punct + mappings.extra_punct))
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    return text


def spacing_some_connect_words(text):
    mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']
    mis_connect_re  = re.compile('(%s)' % '|'.join(mis_connect_list))

    ori = text
    for error in mappings.mis_spell_mapping:
        if error in text:
            text = text.replace(error, mappings.mis_spell_mapping[error])

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')
    text = remove_spaces(text)
    return text


def clean_bad_case_words(text):
    for bad_word in mappings.bad_case_words:
        if bad_word in text:
             text = text.replace(bad_word, mappings.bad_case_words[bad_word])
    return text


def clean_repeat_words(text):
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text) #this one is causing few issues(fixed via monkey patching in other dicts for now), need to check it..
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    text = re.sub(r"(-+|\.+)", " ", text)
    return text


def correct_spelling(text):
    for word in mappings.mispell_dict.keys():
        if word in text:
            text = text.replace(word, mappings.mispell_dict[word])
    return text


def correct_contraction(text):
    for word in mappings.contraction_mapping.keys():
        if word in text:
            text = text.replace(word, mappings.contraction_mapping[word])
    return text


def clean_text(text):
    text = str(text).replace(' s ','').replace('…', ' ').replace('—','-').replace('•°•°•','') #should be broken down to regetexts (lazy to do it haha)
    for punct in "/-'":
        if punct in text:
            text = text.replace(punct, ' ')
    for punct in '&':
        if punct in text:
            text = text.replace(punct, f' {punct} ')
    for punct in '?!-,"#$%\'()*+-/:;<=>@[\\]^_`{|}~–—✰«»§✈➤›☭✔½☺😏🤣😢😁🙄😃😄😊😜😎😆💙👍🤔😅😡▀▄·―═►♥▬' + '“”’': #if we add . here then all the WEBPAGE LINKS WILL VANISH WE DON'T WANT THAT
        if punct in text: #can be used a FE for emojis but here we are just removing them..
            text = text.replace(punct, '')
    for punct in '.•': #hence here it is
        if punct in text:
            text = text.replace(punct, f' ')

    #can be improved more.....

    # text = re.sub(r'[\text00-\text1f\text7f-\text9f\textad]', '', text)
    text = re.sub(r'(\d+)(e)(\d+)',r'\g<1> \g<3>', text) #is a dup from above cell...
    text = re.sub(r"(-+|\.+)\s?", "  ", text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub(r'ᴵ+', '', text)

    text = re.sub(r'(been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|their|has|have|the|be|that|not|was|he|just|they|who)(how)', '\g<1> \g<2>', text)
    #fitexting words like hehow, therehow, hishow... (for some reason they are getting added, some flaw in preproceesing..)
    #the last regetext ia a temp fitext mainly and shouldn't be there
    return text


def preprocess_text(text):
    text = remove_spaces(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = clean_rare_words(text)
    text = clean_misspell(text)
    text = spacing_punctuation(text)
    text = spacing_some_connect_words(text)
    text = clean_bad_case_words(text)
    text = clean_repeat_words(text)
    text = correct_contraction(text)
    text = correct_spelling(text)
    text = clean_text(text)
    text = remove_spaces(text)
    return text
