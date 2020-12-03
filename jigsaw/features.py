def load_clean_words(clean_words_path):
    clean_word_dict = {}
    with open(clean_words_path, 'r', encoding='utf-8') as cl:
        for line in cl:
            line = line.strip('\n')
            typo, correct = line.split(',')
            clean_word_dict[typo] = correct
    return clean_word_dict



def add_features(df):

    df['comment_text'] = df['comment_text'].apply(lambda x: str(x))
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                    axis=1)
    df['num_words'] = df.comment_text.str.count('\S+')
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df
