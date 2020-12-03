def handle_pooltype(df_train):
    # => reduce test score from 0.0646261 to 0.0646457
    df_train["pooltype"] = 0
    df_train["pooltype"].ix[df_train["pooltypeid10"] == 1] = 10
    df_train["pooltype"].ix[df_train["pooltypeid2"] == 1] = 2
    df_train["pooltype"].ix[df_train["pooltypeid7"] == 1] = 7
    df_train.drop(["pooltypeid10", "pooltypeid2", "pooltypeid7"], axis=1)
    return df_train